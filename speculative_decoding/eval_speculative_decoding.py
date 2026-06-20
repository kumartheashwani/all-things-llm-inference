#!/usr/bin/env python3
"""
eval_speculative_decoding.py — Learn speculative decoding by seeing it in action.

  Draft model  : distilgpt2   (82M params)  — cheap speculator
  Target model : gpt2-medium  (345M params) — authoritative distribution

HOW IT WORKS
───────────────────────────────────────────────────────────────────────────────
Standard autoregressive generation calls the target model ONCE PER TOKEN.
For 100 tokens → 100 sequential target-model forward passes.

Speculative decoding gets more tokens per target-model call:

  1. DRAFT PHASE   Draft model autoregressively generates K candidate tokens.
                   This is fast because the draft model is much smaller.

  2. VERIFY PHASE  Target model runs ONE forward pass over all K candidates.
                   Because modern accelerators parallelize across sequence length,
                   this costs roughly the same as a single-token forward pass.

  3. ACCEPT/REJECT For each candidate token t_i:
                     Let  q(t) = draft probability,  p(t) = target probability
                     α = min(1, p(t_i) / q(t_i))          ← acceptance probability
                     Draw u ~ Uniform(0,1)
                     If u < α  → accept t_i                ← output matches target
                     Else      → resample from max(0, p−q) normalized, stop accepting

  4. BONUS TOKEN   If all K tokens are accepted, sample one free bonus token
                   from the target's last logit position.

  OUTPUT DISTRIBUTION IS IDENTICAL TO THE TARGET MODEL — this is the
  mathematical guarantee of rejection sampling. Speculative decoding is
  lossless; it trades draft-model compute for target-model compute.

  SPEEDUP = avg tokens per target call = K × acceptance_rate + 1
  With K=4 and 80% acceptance rate → ~4.2 tokens per target call → 4.2× speedup.

USAGE
───────────────────────────────────────────────────────────────────────────────
  pip install transformers torch
  python eval_speculative_decoding.py                      # default: educational trace + benchmark
  python eval_speculative_decoding.py --trace-only         # just show accept/reject steps
  python eval_speculative_decoding.py --lookahead 8 --max-new-tokens 150
  python eval_speculative_decoding.py --target gpt2-large --draft gpt2

WHEN TO USE IN PRODUCTION
───────────────────────────────────────────────────────────────────────────────
  ✓ Single-stream generation (batch=1) where the GPU is underutilised
  ✓ Predictable outputs: code, structured data, factual text → high acceptance
  ✓ Latency-sensitive serving where you want lower time-to-complete
  ✗ Large-batch inference — GPU is already saturated, no room for draft model
  ✗ High-temperature creative text — low acceptance rate, less gain
"""

import argparse
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_DRAFT  = "distilgpt2"    # 82M params
DEFAULT_TARGET = "gpt2-medium"  # 345M params
DEFAULT_K      = 4              # lookahead (draft tokens per step)
DEFAULT_TOKENS = 80             # new tokens to generate per prompt

TEST_PROMPTS = [
    "The speed of light in vacuum is approximately 299,792 kilometres per second, which means",
    "def merge_sort(arr):\n    \"\"\"Sort array using merge sort.\"\"\"\n    if len(arr) <= 1:\n        return arr\n    ",
    "In the field of natural language processing, transformer architectures have revolutionised",
    "Once upon a time there was a small robot who dreamed of becoming an astronaut. Every night",
    "The key difference between TCP and UDP network protocols is that TCP",
]

W = 72  # line width for headers


# ─────────────────────────────────────────────────────────────────────────────
#  CORE ALGORITHM  (no KV cache — so the algorithm is maximally readable)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def speculative_decode(
    target_model: AutoModelForCausalLM,
    draft_model:  AutoModelForCausalLM,
    tokenizer:    AutoTokenizer,
    input_ids:    torch.Tensor,
    max_new_tokens: int = 40,
    K: int = 4,
    temperature: float = 0.8,
    verbose: bool = False,
) -> tuple:
    """
    Speculative decoding implemented from scratch.

    Intentionally avoids KV-cache management so the algorithm reads cleanly.
    In production (HuggingFace generate with assistant_model=) KV cache is
    used and the wall-clock speedup is real.

    Returns: (output_ids, stats_dict)
    """
    device     = input_ids.device
    generated  = input_ids.clone()
    prompt_len = input_ids.shape[1]

    total_steps    = 0   # number of target model forward passes
    total_drafted  = 0
    total_accepted = 0

    while generated.shape[1] - prompt_len < max_new_tokens:
        n         = generated.shape[1]
        remaining = max_new_tokens - (n - prompt_len)
        k         = min(K, remaining)

        # ── 1. DRAFT PHASE ──────────────────────────────────────────────────
        # Draft model autoregressively generates K candidate tokens.
        # We store the full probability distribution at each position so we can
        # compute acceptance probabilities during the verify phase.
        draft_tokens = []   # list of (1,1) tensors
        draft_probs  = []   # list of (vocab,) tensors — full distribution q(t)

        ctx = generated
        for _ in range(k):
            logits = draft_model(ctx).logits[:, -1, :]
            probs  = F.softmax(logits / temperature, dim=-1)
            tok    = torch.multinomial(probs, 1)
            draft_tokens.append(tok)
            draft_probs.append(probs[0])          # (vocab,)
            ctx = torch.cat([ctx, tok], dim=1)

        # ── 2. VERIFY PHASE ─────────────────────────────────────────────────
        # Target model runs ONE forward pass over context + K draft tokens.
        # This produces K+1 logit vectors (positions n..n+K) in a single call.
        # Cost ≈ single-token target forward pass (sequence parallelism).
        verify_ids    = torch.cat([generated] + draft_tokens, dim=1)  # (1, n+k)
        target_logits = target_model(verify_ids).logits                # (1, n+k, vocab)
        total_steps  += 1

        # ── 3. ACCEPT / REJECT ──────────────────────────────────────────────
        # For each draft token, decide to accept or resample using rejection sampling.
        # Indexing: logit at position j predicts token j+1, so
        #   target distribution for draft token i (at position n+i) = logits[:, n-1+i, :]
        n_accepted     = 0
        correction_tok = None
        outcomes       = []   # for verbose printing

        for i in range(k):
            t_i = draft_tokens[i][0, 0]                       # scalar token id
            q_i = draft_probs[i]                               # (vocab,) draft dist
            p_i = F.softmax(
                target_logits[0, n - 1 + i, :] / temperature, dim=-1
            )                                                  # (vocab,) target dist

            # Acceptance probability — the min(1, p/q) is the rejection-sampling bound
            alpha  = min(1.0, (p_i[t_i] / (q_i[t_i] + 1e-10)).item())
            accept = torch.rand(1).item() < alpha

            if accept:
                n_accepted += 1
                outcomes.append(("ACCEPT", t_i.item(), alpha))
                if n_accepted == k:
                    # All K draft tokens accepted → grab a free bonus token from
                    # the target's logit at position n+K (cost already paid above).
                    bonus_p = F.softmax(
                        target_logits[0, n - 1 + k, :] / temperature, dim=-1
                    )
                    correction_tok = torch.multinomial(bonus_p.unsqueeze(0), 1)
                    outcomes.append(("BONUS", correction_tok[0, 0].item(), 1.0))
            else:
                # Reject: resample from the adjusted distribution max(0, p−q)/Z.
                # This ensures the overall output matches the target distribution.
                adjusted = F.relu(p_i - q_i)
                s        = adjusted.sum()
                adjusted = adjusted / s if s > 0 else p_i
                correction_tok = torch.multinomial(adjusted.unsqueeze(0), 1)
                outcomes.append(("REJECT", t_i.item(), alpha))
                break  # discard remaining draft tokens after first rejection

        # ── 4. APPEND ───────────────────────────────────────────────────────
        to_add = draft_tokens[:n_accepted] + ([correction_tok] if correction_tok is not None else [])
        if to_add:
            generated = torch.cat([generated] + to_add, dim=1)

        total_drafted  += k
        total_accepted += n_accepted

        if verbose:
            tokens_added = n_accepted + (1 if correction_tok is not None else 0)
            draft_strs   = [tokenizer.decode([draft_tokens[i][0, 0].item()]) for i in range(k)]
            parts = []
            for o_type, tok_id, alpha in outcomes:
                s = tokenizer.decode([tok_id])
                if o_type == "ACCEPT":
                    parts.append(f"\033[32m✓{s!r}(α={alpha:.2f})\033[0m")
                elif o_type == "BONUS":
                    parts.append(f"\033[34m★{s!r}\033[0m")
                else:
                    parts.append(f"\033[31m✗{s!r}(α={alpha:.2f})\033[0m")
            print(f"    step {total_steps:2d} | draft=[{', '.join(repr(t) for t in draft_strs)}]"
                  f" → {' '.join(parts)}  (+{tokens_added})")

        if generated[0, -1].item() == (tokenizer.eos_token_id or -1):
            break

    tokens_out = generated.shape[1] - prompt_len
    ar         = total_accepted / total_drafted if total_drafted else 0.0
    stats = {
        "acceptance_rate":           ar,
        "total_target_calls":        total_steps,
        "tokens_generated":          tokens_out,
        "avg_tokens_per_target_call": tokens_out / max(total_steps, 1),
    }
    return generated, stats


# ─────────────────────────────────────────────────────────────────────────────
#  BENCHMARK HELPERS (HuggingFace generate — uses KV cache, production speed)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchResult:
    method:  str
    new_tok: int
    elapsed: float
    tps:     float   # tokens per second


def bench_autoregressive(model, tokenizer, input_ids, max_new_tokens) -> BenchResult:
    t0  = time.perf_counter()
    out = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    el  = time.perf_counter() - t0
    nt  = out.shape[1] - input_ids.shape[1]
    return BenchResult("Autoregressive (greedy)", nt, el, nt / el)


def bench_speculative_hf(target, draft, tokenizer, input_ids, max_new_tokens) -> BenchResult:
    """
    Uses HuggingFace's built-in speculative decoding via assistant_model=.
    Internally uses KV cache and adaptive K — this is what you'd ship in production.
    """
    t0 = time.perf_counter()
    out = target.generate(
        input_ids,
        assistant_model=draft,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    el = time.perf_counter() - t0
    nt = out.shape[1] - input_ids.shape[1]
    return BenchResult("Speculative (HF built-in)", nt, el, nt / el)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def hr(char="─", n=W):
    print(char * n)


def header(title, char="═"):
    print()
    hr(char)
    print(title)
    hr(char)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draft",          default=DEFAULT_DRAFT,  help=f"Draft model (default: {DEFAULT_DRAFT})")
    parser.add_argument("--target",         default=DEFAULT_TARGET, help=f"Target model (default: {DEFAULT_TARGET})")
    parser.add_argument("--lookahead", "-k", type=int, default=DEFAULT_K, help=f"K draft tokens per step (default: {DEFAULT_K})")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_TOKENS, dest="max_new_tokens")
    parser.add_argument("--temperature",    type=float, default=0.8)
    parser.add_argument("--trace-only",     action="store_true", help="Run educational trace only, skip HF benchmark")
    args = parser.parse_args()

    # ── Device ───────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    header("SPECULATIVE DECODING — EVALUATION SCRIPT")
    print(f"  Draft model  : {args.draft}")
    print(f"  Target model : {args.target}")
    print(f"  Lookahead K  : {args.lookahead}")
    print(f"  Max tokens   : {args.max_new_tokens}")
    print(f"  Temperature  : {args.temperature}")
    print(f"  Device       : {device}")

    # ── Load models ──────────────────────────────────────────────────────────
    header("LOADING MODELS")
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading {args.draft} (draft)...", end=" ", flush=True)
    draft_model = AutoModelForCausalLM.from_pretrained(args.draft, torch_dtype=torch.float32).to(device)
    draft_model.eval()
    d_params = sum(p.numel() for p in draft_model.parameters()) / 1e6
    print(f"done  ({d_params:.0f}M params)")

    print(f"  Loading {args.target} (target)...", end=" ", flush=True)
    target_model = AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=torch.float32).to(device)
    target_model.eval()
    t_params = sum(p.numel() for p in target_model.parameters()) / 1e6
    print(f"done  ({t_params:.0f}M params)")
    print(f"\n  Draft is {t_params/d_params:.1f}× smaller than target — more compression → more speedup potential")

    # ── Educational trace ─────────────────────────────────────────────────────
    header("EDUCATIONAL TRACE — step-by-step accept/reject decisions")
    print(f"  Colour key:  \033[32m✓ ACCEPT\033[0m   \033[31m✗ REJECT (resample)\033[0m   \033[34m★ BONUS\033[0m")
    print(f"  (No KV cache here — makes algorithm readable; timing not meaningful)")
    hr()

    trace_prompt = TEST_PROMPTS[0]
    print(f"Prompt: {trace_prompt!r}\n")

    trace_ids = tokenizer(trace_prompt, return_tensors="pt").input_ids.to(device)
    out_ids, stats = speculative_decode(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        input_ids=trace_ids,
        max_new_tokens=min(40, args.max_new_tokens),
        K=args.lookahead,
        temperature=args.temperature,
        verbose=True,
    )

    output_text = tokenizer.decode(out_ids[0, trace_ids.shape[1]:], skip_special_tokens=True)
    print()
    print(f"Generated text : {output_text!r}")
    print()
    hr()
    print(f"  Acceptance rate          : {stats['acceptance_rate']:.1%}")
    print(f"  Target model calls       : {stats['total_target_calls']}")
    print(f"  Tokens generated         : {stats['tokens_generated']}")
    print(f"  Avg tokens / target call : {stats['avg_tokens_per_target_call']:.2f}  (baseline autoregressive = 1.0)")
    print(f"  Theoretical speedup      : ≈ {stats['avg_tokens_per_target_call']:.1f}× (realised with KV cache)")
    print()
    print("  FORMULA: avg_tokens_per_target_call = K × acceptance_rate + 1")
    print(f"           {args.lookahead} × {stats['acceptance_rate']:.2f} + 1 = {args.lookahead * stats['acceptance_rate'] + 1:.2f}")

    # Run trace on all prompts to measure acceptance rate variance
    header("ACCEPTANCE RATE BY PROMPT TYPE")
    print("  (acceptance rate varies by content — code and factual text score higher)\n")
    all_ar = []
    for i, prompt in enumerate(TEST_PROMPTS):
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        _, s = speculative_decode(
            target_model, draft_model, tokenizer, ids,
            max_new_tokens=min(40, args.max_new_tokens),
            K=args.lookahead, temperature=args.temperature, verbose=False,
        )
        ar = s["acceptance_rate"]
        all_ar.append(ar)
        bar = "█" * int(ar * 30)
        label = prompt[:55].replace("\n", "↵")
        print(f"  [{i+1}] {ar:5.1%}  {bar:<30}  {label!r}...")

    print(f"\n  Average acceptance rate: {sum(all_ar)/len(all_ar):.1%}")

    if args.trace_only:
        _print_integration_snippet(args)
        return

    # ── HF Benchmark ─────────────────────────────────────────────────────────
    header("BENCHMARK — HuggingFace generate() with KV cache (production speed)")
    print(f"  Greedy decoding | {args.max_new_tokens} new tokens per prompt | {len(TEST_PROMPTS)} prompts")
    print(f"  Both methods produce identical output (speculative is lossless)")
    hr()

    ar_results   = []
    spec_results = []

    for i, prompt in enumerate(TEST_PROMPTS):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        label     = prompt[:50].replace("\n", "↵")
        print(f"  [{i+1}/{len(TEST_PROMPTS)}] {label!r}...")

        # Warm-up the first prompt to avoid cold-start bias
        if i == 0:
            _ = target_model.generate(input_ids, max_new_tokens=5, do_sample=False)

        ar   = bench_autoregressive(target_model, tokenizer, input_ids, args.max_new_tokens)
        spec = bench_speculative_hf(target_model, draft_model, tokenizer, input_ids, args.max_new_tokens)

        ar_results.append(ar)
        spec_results.append(spec)

        speedup = spec.tps / ar.tps
        print(f"         Autoregressive : {ar.tps:6.1f} tok/s  ({ar.elapsed:.2f}s)")
        print(f"         Speculative    : {spec.tps:6.1f} tok/s  ({spec.elapsed:.2f}s)  speedup: {speedup:.2f}×")

    # ── Summary ───────────────────────────────────────────────────────────────
    header("SUMMARY TABLE")

    avg_ar_tps   = sum(r.tps     for r in ar_results)   / len(ar_results)
    avg_spec_tps = sum(r.tps     for r in spec_results) / len(spec_results)
    avg_ar_el    = sum(r.elapsed for r in ar_results)   / len(ar_results)
    avg_spec_el  = sum(r.elapsed for r in spec_results) / len(spec_results)
    avg_speedup  = avg_spec_tps / avg_ar_tps

    col = "  {:<30}  {:>10}  {:>12}  {:>8}"
    print(col.format("Method", "tok/s (avg)", "time/prompt", "speedup"))
    hr()
    print(col.format("Autoregressive (greedy)", f"{avg_ar_tps:.1f}", f"{avg_ar_el:.2f}s", "1.00×"))
    print(col.format(f"Speculative K={args.lookahead} (HF)", f"{avg_spec_tps:.1f}", f"{avg_spec_el:.2f}s", f"{avg_speedup:.2f}×"))
    hr()
    print(f"\n  Speedup on {device}: {avg_speedup:.2f}×   |   avg acceptance rate: {sum(all_ar)/len(all_ar):.1%}")

    _print_integration_snippet(args)


def _print_integration_snippet(args):
    header("INTEGRATION SNIPPET  (drop into your inference system)")
    print(f"""
  from transformers import AutoModelForCausalLM, AutoTokenizer

  draft_model  = AutoModelForCausalLM.from_pretrained("{args.draft}").to(device)
  target_model = AutoModelForCausalLM.from_pretrained("{args.target}").to(device)
  tokenizer    = AutoTokenizer.from_pretrained("{args.target}")

  input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

  # ← This ONE change adds speculative decoding to any existing generate() call
  output = target_model.generate(
      input_ids,
      assistant_model=draft_model,   # ← the only change needed
      max_new_tokens=200,
  )
  text = tokenizer.decode(output[0], skip_special_tokens=True)

  # TIPS
  # ─────────────────────────────────────────────────────────────────────────
  # • Draft model must share the same tokenizer vocabulary as target.
  # • Larger draft-to-target size ratio = more speedup potential.
  # • Works with both greedy (do_sample=False) and sampling (do_sample=True).
  # • HF adaptively tunes K — no manual lookahead tuning needed.
  # • For vLLM/TGI/TensorRT-LLM: each has its own speculative decoding API
  #   but the concept and draft model choice are identical.
""")


if __name__ == "__main__":
    main()
