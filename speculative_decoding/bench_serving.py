#!/usr/bin/env python3
"""
bench_serving.py — Benchmark vLLM serving with and without speculative decoding.

Starts two vLLM server instances sequentially, sends identical workloads against
each, and reports a side-by-side comparison of:

  TTFT         Time To First Token (p50, p95) — latency until first output word
  ITL          Inter-Token Latency (p50, p95) — speed of subsequent tokens
  Throughput   Output tokens / second

RECOMMENDED MODEL PAIRS BY GPU
───────────────────────────────────────────────────────────────────────────────
  T4  16GB  (Colab free)  : --target facebook/opt-1.3b   --draft facebook/opt-125m
  A10G 24GB (RunPod)      : --target Qwen/Qwen2.5-7B-Instruct --draft Qwen/Qwen2.5-0.5B-Instruct
  A100 40GB (RunPod/Colab): --target meta-llama/Llama-3.1-8B-Instruct --draft meta-llama/Llama-3.2-1B-Instruct
  A100 80GB               : --target meta-llama/Llama-3.1-70B-Instruct --draft meta-llama/Llama-3.2-1B-Instruct --tp 2

INSTALL
───────────────────────────────────────────────────────────────────────────────
  pip install vllm openai requests

USAGE
───────────────────────────────────────────────────────────────────────────────
  # Smallest — works on Colab T4 free tier
  python bench_serving.py --target facebook/opt-1.3b --draft facebook/opt-125m

  # Llama (needs ~20GB VRAM — RunPod A10G or better, requires HF token)
  export HF_TOKEN=hf_xxx
  python bench_serving.py \\
      --target meta-llama/Llama-3.1-8B-Instruct \\
      --draft  meta-llama/Llama-3.2-1B-Instruct

  # No draft model (ngram speculative decoding — works with any model, no extra VRAM)
  python bench_serving.py --target facebook/opt-1.3b --ngram

  # Just benchmark a server you started separately
  python bench_serving.py --external --port 8000 --target facebook/opt-1.3b

WHY SERVING METRICS MATTER (beyond model-level benchmarks)
───────────────────────────────────────────────────────────────────────────────
  Model-level benchmarks (like eval_speculative_decoding.py) measure pure
  algorithmic speed with no network stack, batching, or scheduler overhead.
  Serving benchmarks measure what users actually experience:

    • TTFT includes time for request queuing, tokenisation, prefill, and
      first-token generation — speculative decoding reduces this by generating
      K+1 tokens per target-model call.

    • ITL reflects the decode phase speed — the primary target of speculative
      decoding. Lower ITL = faster perceived "typing" effect for the user.

    • Throughput matters for multi-user deployments. Note: speculative decoding
      helps most at batch=1 (single concurrent user). At large batch sizes the
      GPU is already saturated and the benefit diminishes.

TOOLS COMPARISON
───────────────────────────────────────────────────────────────────────────────
  This script (custom)    : shows algorithm internals + TTFT/ITL/throughput.
                            No extra deps beyond openai + requests.
  LLMPerf (Anyscale)      : production-grade, concurrent load testing, Ray-based.
                            Best for multi-user throughput curves.
  vLLM benchmark_serving  : built-in, integrates with ShareGPT datasets.
                            Best for apples-to-apples vLLM comparisons.
  Locust / k6             : HTTP load testing. Useful if you run vLLM behind nginx.

  Start with this script. Graduate to LLMPerf when you need concurrency curves.
"""

import argparse
import os
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

BENCHMARK_PROMPTS = [
    "Explain how gradient descent works in machine learning.",
    "Write a Python function to implement binary search on a sorted list.",
    "What are the key architectural differences between BERT and GPT?",
    "Describe how the transformer self-attention mechanism works step by step.",
    "Write a SQL query to find the top 5 customers by total order value.",
    "Explain the CAP theorem in distributed systems with a real-world example.",
    "What is the difference between a process and a thread?",
    "Write a recursive Python function to compute the nth Fibonacci number.",
    "Describe the backpropagation algorithm and how gradients flow through layers.",
    "What are the SOLID principles in object-oriented software design?",
    "Explain how a hash table handles collisions using chaining.",
    "Write a Python class implementing a min-heap data structure.",
]

W = 72


def hr(char="─"):
    print(char * W)


# ─────────────────────────────────────────────────────────────────────────────
#  DEPENDENCY CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def check_deps():
    missing = []
    for pkg in ["openai", "requests"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing: pip install {' '.join(missing)}")
        sys.exit(1)


def check_vllm():
    try:
        import vllm  # noqa: F401
    except ImportError:
        print("vLLM not installed. Run: pip install vllm")
        print("Note: vLLM requires a CUDA GPU. Install on RunPod or Colab.")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RequestMetrics:
    ttft_ms: float
    total_time_s: float
    output_tokens: int
    tps: float
    itl_ms_list: list = field(default_factory=list)


@dataclass
class BenchmarkMetrics:
    label: str
    ttft_p50_ms: float
    ttft_p95_ms: float
    itl_p50_ms: float
    itl_p95_ms: float
    throughput_tps: float
    n_requests: int
    n_failed: int


def percentile(lst, p):
    if not lst:
        return 0.0
    s = sorted(lst)
    idx = max(0, min(int(len(s) * p / 100), len(s) - 1))
    return s[idx]


# ─────────────────────────────────────────────────────────────────────────────
#  SINGLE REQUEST MEASUREMENT
# ─────────────────────────────────────────────────────────────────────────────

def measure_request(client, model_name: str, prompt: str, max_tokens: int) -> Optional[RequestMetrics]:
    """
    Send one streaming chat request and measure:
      TTFT  — time from send to first non-empty chunk
      ITL   — list of inter-chunk intervals (proxy for inter-token latency)
      TPS   — output tokens / total request time
    """
    t_start = time.perf_counter()
    t_first = None
    chunk_times = []

    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            stream=True,
            temperature=0,
        )
        for chunk in stream:
            now = time.perf_counter()
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                if t_first is None:
                    t_first = now
                chunk_times.append(now)
    except Exception as e:
        print(f"\n    [WARN] Request error: {e}")
        return None

    t_end = time.perf_counter()

    ttft_ms    = (t_first - t_start) * 1000 if t_first else 0.0
    total_time = t_end - t_start
    n_tokens   = len(chunk_times)
    tps        = n_tokens / total_time if total_time > 0 else 0.0

    itl_list = []
    if len(chunk_times) > 1:
        itl_list = [(chunk_times[i] - chunk_times[i - 1]) * 1000
                    for i in range(1, len(chunk_times))]

    return RequestMetrics(ttft_ms, total_time, n_tokens, tps, itl_list)


# ─────────────────────────────────────────────────────────────────────────────
#  BENCHMARK LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(port: int, model_name: str, label: str,
                  prompts: list, max_tokens: int, n_requests: int) -> BenchmarkMetrics:
    from openai import OpenAI

    client   = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="none")
    all_ttft = []
    all_tps  = []
    all_itl  = []
    n_failed = 0

    # Warm-up: one request not counted in results
    print(f"  Warm-up request...", end=" ", flush=True)
    measure_request(client, model_name, prompts[0], max_tokens=20)
    print("done")

    for i in range(n_requests):
        prompt = prompts[i % len(prompts)]
        sys.stdout.write(f"\r  Request {i + 1:>3}/{n_requests}...")
        sys.stdout.flush()

        m = measure_request(client, model_name, prompt, max_tokens)
        if m and m.output_tokens > 0:
            all_ttft.append(m.ttft_ms)
            all_tps.append(m.tps)
            all_itl.extend(m.itl_ms_list)
        else:
            n_failed += 1

    sys.stdout.write(f"\r  {n_requests} requests complete ({n_failed} failed)      \n")

    return BenchmarkMetrics(
        label=label,
        ttft_p50_ms=percentile(all_ttft, 50),
        ttft_p95_ms=percentile(all_ttft, 95),
        itl_p50_ms=percentile(all_itl, 50) if all_itl else 0.0,
        itl_p95_ms=percentile(all_itl, 95) if all_itl else 0.0,
        throughput_tps=statistics.mean(all_tps) if all_tps else 0.0,
        n_requests=n_requests,
        n_failed=n_failed,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  SERVER LIFECYCLE
# ─────────────────────────────────────────────────────────────────────────────

def build_server_cmd(target_model, port, served_name, extra_args=None, tp=1):
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model",               target_model,
        "--port",                str(port),
        "--served-model-name",   served_name,
        "--max-model-len",       "2048",
        "--gpu-memory-utilization", "0.85",
        "--disable-log-requests",
    ]
    if tp > 1:
        cmd += ["--tensor-parallel-size", str(tp)]
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def start_server(cmd, log_path, env=None):
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env=env or os.environ.copy(),
        preexec_fn=os.setsid,
    )
    return proc, log_fh


def wait_for_server(port: int, timeout: int = 360) -> bool:
    import requests as req

    url   = f"http://localhost:{port}/health"
    start = time.time()
    sys.stdout.write("  Waiting for server")
    sys.stdout.flush()

    while time.time() - start < timeout:
        try:
            r = req.get(url, timeout=2)
            if r.status_code == 200:
                elapsed = time.time() - start
                print(f"  ready in {elapsed:.0f}s")
                return True
        except Exception:
            pass
        time.sleep(4)
        sys.stdout.write(".")
        sys.stdout.flush()

    print("  TIMED OUT")
    return False


def stop_server(proc, log_fh=None):
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=20)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    if log_fh:
        log_fh.close()
    time.sleep(3)  # allow port to free


# ─────────────────────────────────────────────────────────────────────────────
#  RESULTS PRINTING
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison(baseline: BenchmarkMetrics, spec: BenchmarkMetrics):
    col = "  {:<26}  {:>14}  {:>14}  {:>10}"

    print()
    hr("═")
    print("RESULTS")
    hr("═")
    print(col.format("Metric", baseline.label, spec.label, "Change"))
    hr()

    def row(label, base_val, spec_val, fmt="{:.1f}", lower_is_better=False):
        ratio = (spec_val / base_val) if base_val > 0 else 1.0
        if lower_is_better:
            change = f"{'▼' if ratio < 0.95 else ('▲' if ratio > 1.05 else '≈')} {1/ratio:.2f}×"
        else:
            change = f"{'▲' if ratio > 1.05 else ('▼' if ratio < 0.95 else '≈')} {ratio:.2f}×"
        print(col.format(label, fmt.format(base_val), fmt.format(spec_val), change))

    row("TTFT p50 (ms)",      baseline.ttft_p50_ms, spec.ttft_p50_ms, lower_is_better=True)
    row("TTFT p95 (ms)",      baseline.ttft_p95_ms, spec.ttft_p95_ms, lower_is_better=True)
    row("ITL  p50 (ms)",      baseline.itl_p50_ms,  spec.itl_p50_ms,  lower_is_better=True)
    row("ITL  p95 (ms)",      baseline.itl_p95_ms,  spec.itl_p95_ms,  lower_is_better=True)
    row("Throughput (tok/s)", baseline.throughput_tps, spec.throughput_tps, fmt="{:.2f}")

    hr()
    speedup = spec.throughput_tps / baseline.throughput_tps if baseline.throughput_tps else 1.0
    print(f"\n  Throughput speedup: {speedup:.2f}×")
    print(f"  Requests (baseline / spec): {baseline.n_requests} / {spec.n_requests}")
    print(f"  Failed   (baseline / spec): {baseline.n_failed}  / {spec.n_failed}")

    print()
    hr()
    print("METRIC GLOSSARY")
    hr()
    print("""
  TTFT  Time To First Token — latency a user experiences before seeing any
        output. Includes request queuing, prefill, and first decode step.
        Speculative decoding reduces this because K+1 tokens come from one
        target-model call.

  ITL   Inter-Token Latency — milliseconds between consecutive streaming
        tokens. Lower ITL = faster "typing" feel for the user. This is the
        primary metric that speculative decoding optimises.

  Throughput  Output tokens per second for a single request stream (batch=1).
              Speculative decoding helps most here. At large concurrent batch
              sizes the GPU is already saturated and the benefit shrinks.

  ▲  improvement   ▼  regression   ≈  no significant change (<5%)
""")


def print_single(m: BenchmarkMetrics):
    hr("═")
    print(f"RESULTS — {m.label}")
    hr()
    print(f"  TTFT p50       : {m.ttft_p50_ms:.1f} ms")
    print(f"  TTFT p95       : {m.ttft_p95_ms:.1f} ms")
    print(f"  ITL p50        : {m.itl_p50_ms:.1f} ms")
    print(f"  ITL p95        : {m.itl_p95_ms:.1f} ms")
    print(f"  Throughput     : {m.throughput_tps:.2f} tok/s")
    print(f"  Requests       : {m.n_requests}  (failed: {m.n_failed})")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM speculative decoding vs autoregressive serving.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--target",           default="facebook/opt-1.3b",
                        help="Target (large) model  (default: facebook/opt-1.3b)")
    parser.add_argument("--draft",            default="facebook/opt-125m",
                        help="Draft (small) model   (default: facebook/opt-125m)")
    parser.add_argument("--num-spec-tokens",  type=int, default=5,  dest="K",
                        help="K draft tokens per step (default: 5)")
    parser.add_argument("--port",             type=int, default=8000)
    parser.add_argument("--n-requests",       type=int, default=20, dest="n_requests",
                        help="Benchmark requests per mode (default: 20)")
    parser.add_argument("--max-tokens",       type=int, default=200, dest="max_tokens")
    parser.add_argument("--tp",               type=int, default=1,
                        help="Tensor parallel size for multi-GPU (default: 1)")
    parser.add_argument("--hf-token",         default=os.environ.get("HF_TOKEN"),
                        dest="hf_token", help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--ngram",            action="store_true",
                        help="Use ngram speculative decoding (no draft model, any target model)")
    parser.add_argument("--baseline-only",    action="store_true", dest="baseline_only",
                        help="Only benchmark baseline, skip speculative run")
    parser.add_argument("--external",         action="store_true",
                        help="Connect to an already-running vLLM server (do not start one)")
    parser.add_argument("--log-dir",          default="logs", dest="log_dir")
    args = parser.parse_args()

    check_deps()
    if not args.external:
        check_vllm()

    os.makedirs(args.log_dir, exist_ok=True)

    env = os.environ.copy()
    if args.hf_token:
        env["HF_TOKEN"]                  = args.hf_token
        env["HUGGING_FACE_HUB_TOKEN"]    = args.hf_token

    spec_label = f"Speculative (ngram K={args.K})" if args.ngram else f"Speculative (draft K={args.K})"

    print()
    hr("═")
    print("SERVING BENCHMARK — Speculative Decoding vs Autoregressive")
    hr("═")
    print(f"  Target model       : {args.target}")
    if not args.ngram:
        print(f"  Draft model        : {args.draft}")
    else:
        print(f"  Draft method       : ngram (no draft model required)")
    print(f"  Speculative K      : {args.K}")
    print(f"  Requests per mode  : {args.n_requests}")
    print(f"  Max output tokens  : {args.max_tokens}")
    print(f"  Tensor parallel    : {args.tp}")
    print(f"  Port               : {args.port}")
    print(f"  Logs               : {args.log_dir}/")

    baseline_metrics = None
    spec_metrics     = None

    # ── EXTERNAL mode: server already running ─────────────────────────────────
    if args.external:
        print()
        hr()
        print("EXTERNAL mode — connecting to existing server")
        hr()
        print(f"  Expecting server at http://localhost:{args.port}")
        print(f"  Model name used for API calls: {args.target.split('/')[-1]}")

        served_name = args.target.split("/")[-1]
        baseline_metrics = run_benchmark(
            port=args.port, model_name=served_name, label="External server",
            prompts=BENCHMARK_PROMPTS, max_tokens=args.max_tokens, n_requests=args.n_requests,
        )
        print_single(baseline_metrics)
        return

    # ── 1. BASELINE ────────────────────────────────────────────────────────────
    print()
    hr()
    print("[1/2] BASELINE — autoregressive (no speculative decoding)")
    hr()

    baseline_log = os.path.join(args.log_dir, "vllm_baseline.log")
    cmd = build_server_cmd(args.target, args.port, "baseline", tp=args.tp)
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Log    : {baseline_log}")

    proc, log_fh = start_server(cmd, baseline_log, env=env)
    try:
        if wait_for_server(args.port):
            baseline_metrics = run_benchmark(
                port=args.port, model_name="baseline", label="Autoregressive",
                prompts=BENCHMARK_PROMPTS, max_tokens=args.max_tokens, n_requests=args.n_requests,
            )
        else:
            print(f"  Server failed to start. See {baseline_log} for errors.")
    finally:
        stop_server(proc, log_fh)

    if args.baseline_only or baseline_metrics is None:
        if baseline_metrics:
            print_single(baseline_metrics)
        return

    # ── 2. SPECULATIVE ─────────────────────────────────────────────────────────
    print()
    hr()
    print(f"[2/2] SPECULATIVE — {spec_label}")
    hr()

    if args.ngram:
        # Ngram speculative decoding — no draft model required, uses local context
        spec_extra = [
            "--speculative-model",      "[ngram]",
            "--num-speculative-tokens", str(args.K),
            "--ngram-prompt-lookup-max", "4",
        ]
    else:
        spec_extra = [
            "--speculative-model",      args.draft,
            "--num-speculative-tokens", str(args.K),
        ]

    spec_log = os.path.join(args.log_dir, "vllm_speculative.log")
    cmd = build_server_cmd(args.target, args.port, "speculative", extra_args=spec_extra, tp=args.tp)
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Log    : {spec_log}")

    proc, log_fh = start_server(cmd, spec_log, env=env)
    try:
        if wait_for_server(args.port):
            spec_metrics = run_benchmark(
                port=args.port, model_name="speculative", label=spec_label,
                prompts=BENCHMARK_PROMPTS, max_tokens=args.max_tokens, n_requests=args.n_requests,
            )
        else:
            print(f"  Server failed to start. See {spec_log} for errors.")
    finally:
        stop_server(proc, log_fh)

    # ── RESULTS ────────────────────────────────────────────────────────────────
    if baseline_metrics and spec_metrics:
        print_comparison(baseline_metrics, spec_metrics)
    elif baseline_metrics:
        print_single(baseline_metrics)
        print("\n[WARN] Speculative server failed — only baseline results available.")

    print()
    hr("═")
    print("NEXT STEPS")
    hr()
    print("""
  1. Try larger K (--num-spec-tokens 8) — more tokens per target call,
     but diminishing returns if acceptance rate drops.

  2. Test with concurrent requests to see how speedup changes under load:
       pip install llmperf
       python -m llmperf.ray_llm_throughput_benchmark \\
           --model baseline --num-concurrent-requests 4 \\
           --llm-api openai --base-url http://localhost:8000

  3. Plot TTFT vs throughput tradeoff across different K values to find
     the sweet spot for your workload.

  4. Compare ngram vs draft-model speculative decoding:
       python bench_serving.py --ngram          # no extra VRAM, ~1.3x speedup
       python bench_serving.py --draft <model>  # extra VRAM, ~1.5-2.5x speedup
""")


if __name__ == "__main__":
    main()
