#!/usr/bin/env python3
"""
genai_perf_sweep.py — Concurrency sweep: baseline vs speculative decoding via GenAI-Perf.

Starts two vLLM servers sequentially, runs NVIDIA GenAI-Perf at multiple concurrency
levels against each, and writes a CSV + comparison table showing how the speedup
changes as concurrent load grows.

KEY FINDING THIS SCRIPT REVEALS
───────────────────────────────────────────────────────────────────────────────
Speculative decoding helps most at low concurrency (batch=1, 2).
As concurrent requests grow, the GPU is already busy running draft+verify for
many requests simultaneously — the marginal benefit of speculation shrinks.
The CSV + optional chart from this sweep shows the crossover point for your
specific hardware and model pair.

                    Expected shape of results:
  Speedup
    2.5× ┤ ●
    2.0× ┤   ●
    1.5× ┤     ●
    1.0× ┤       ● ─ ─ ─ ─ (baseline)
         └────────────────────────── Concurrency
              1   2   4   8   16

INSTALL
───────────────────────────────────────────────────────────────────────────────
  pip install vllm openai requests
  pip install genai-perf tritonclient[all]    # NVIDIA tools — needs CUDA

USAGE
───────────────────────────────────────────────────────────────────────────────
  # OPT models — runs on T4 16GB (Colab free tier), no HF token needed
  python genai_perf_sweep.py \\
      --target facebook/opt-1.3b \\
      --draft  facebook/opt-125m

  # Llama — needs A10G 24GB+ and HF token
  export HF_TOKEN=hf_xxx
  python genai_perf_sweep.py \\
      --target meta-llama/Llama-3.1-8B-Instruct \\
      --draft  meta-llama/Llama-3.2-1B-Instruct \\
      --concurrencies 1 2 4 8 16

  # Ngram (no draft model, works with any target)
  python genai_perf_sweep.py --target facebook/opt-1.3b --ngram

  # Connect to servers you already started (skip server management)
  python genai_perf_sweep.py \\
      --baseline-url localhost:8000 \\
      --speculative-url localhost:8001 \\
      --target facebook/opt-1.3b

  # Generate matplotlib chart in addition to CSV
  python genai_perf_sweep.py --target facebook/opt-1.3b --draft facebook/opt-125m --plot

WHEN TO USE GENAI-PERF vs OTHER TOOLS
───────────────────────────────────────────────────────────────────────────────
  genai_perf_sweep.py (this)   Concurrency sweep — reveals the speedup curve.
                                Best for: understanding batch-size tradeoffs.
  bench_serving.py             Quick single-stream TTFT/ITL/throughput check.
                                Best for: fast sanity test, no NVIDIA SDK needed.
  eval_speculative_decoding.py Algorithm trace — shows accept/reject per step.
                                Best for: understanding how the algorithm works.
  LLMPerf (Anyscale)           Concurrent throughput + latency distribution.
                                Best for: SLA validation under realistic load.
"""

import argparse
import csv
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

CONCURRENCIES_DEFAULT = [1, 2, 4, 8]
W = 72


def hr(char="─"):
    print(char * W)


# ─────────────────────────────────────────────────────────────────────────────
#  RESULT DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SweepRow:
    concurrency: int
    mode: str              # "baseline" or "speculative"
    ttft_avg_ms: float = 0.0
    ttft_p99_ms: float = 0.0
    itl_avg_ms:  float = 0.0
    itl_p99_ms:  float = 0.0
    output_tps:  float = 0.0
    request_tps: float = 0.0
    parsed_ok:   bool  = False
    log_path:    str   = ""

    def as_dict(self):
        return {
            "concurrency":  self.concurrency,
            "mode":         self.mode,
            "ttft_avg_ms":  round(self.ttft_avg_ms, 2),
            "ttft_p99_ms":  round(self.ttft_p99_ms, 2),
            "itl_avg_ms":   round(self.itl_avg_ms,  2),
            "itl_p99_ms":   round(self.itl_p99_ms,  2),
            "output_tps":   round(self.output_tps,   2),
            "request_tps":  round(self.request_tps,  3),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  DEPENDENCY CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def check_genai_perf() -> str:
    """Return path to genai-perf binary or exit with instructions."""
    cmd = shutil.which("genai-perf")
    if cmd:
        return cmd
    print("genai-perf not found. Install with:")
    print("  pip install genai-perf tritonclient[all]")
    print()
    print("Note: tritonclient requires CUDA libraries. Use on RunPod or Colab GPU.")
    sys.exit(1)


def check_vllm():
    try:
        import vllm  # noqa: F401
    except ImportError:
        print("vLLM not installed. Run: pip install vllm")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
#  METRICS PARSING — stdout (primary) + JSON artifact (secondary)
# ─────────────────────────────────────────────────────────────────────────────

def _numbers_in(text: str):
    """Extract all floats from a string."""
    return [float(x) for x in re.findall(r"[\d]+(?:\.[\d]+)?", text) if float(x) > 0]


def parse_from_stdout(stdout: str) -> dict:
    """
    Parse genai-perf's rich-formatted stdout table.

    genai-perf prints a table like:
      │ Time To First Token    │  234.56ms │ ...  │  305ms │ ...
      │ Inter Token Latency    │   12.34ms │ ...  │   14ms │ ...
      Output token throughput (per sec): 45.23
      Request throughput (per sec): 0.30

    Column order: avg, min, max, p99, p90, p75
    """
    metrics = {}

    for raw_line in stdout.splitlines():
        # Strip unicode box-drawing chars so we get clean text + numbers
        line = re.sub(r"[│┃┡╇└─━┗┘╰╮╭╯┏┓┛━┯┷┼]", " ", raw_line).strip()
        lo   = line.lower()

        if "time to first token" in lo:
            nums = _numbers_in(line)
            if nums:
                metrics["ttft_avg_ms"] = nums[0]
            if len(nums) >= 4:
                metrics["ttft_p99_ms"] = nums[3]   # 4th column = p99

        elif "inter token latency" in lo or "inter-token latency" in lo:
            nums = _numbers_in(line)
            if nums:
                metrics["itl_avg_ms"] = nums[0]
            if len(nums) >= 4:
                metrics["itl_p99_ms"] = nums[3]

        elif "output token throughput" in lo:
            suffix = line.split(":")[-1] if ":" in line else line
            nums = _numbers_in(suffix)
            if nums:
                metrics["output_tps"] = nums[0]

        elif "request throughput" in lo and "output" not in lo:
            suffix = line.split(":")[-1] if ":" in line else line
            nums = _numbers_in(suffix)
            if nums:
                metrics["request_tps"] = nums[0]

    return metrics


def parse_from_artifact(artifact_dir: str) -> dict:
    """
    Try to extract summary metrics from genai-perf's JSON artifact file.
    Returns empty dict if not found — caller falls back to stdout parsing.
    """
    # genai-perf writes a summary JSON alongside profile_export.json
    candidates = glob.glob(
        os.path.join(artifact_dir, "**", "*genai_perf*.json"), recursive=True
    )
    candidates = [f for f in candidates if "profile_export" not in f]
    if not candidates:
        return {}

    try:
        with open(sorted(candidates)[-1]) as fh:
            data = json.load(fh)
    except Exception:
        return {}

    # Some versions wrap everything under "llm_metrics"
    if "llm_metrics" in data:
        data = data["llm_metrics"]

    key_map = {
        "time_to_first_token":     ("ttft_avg_ms", "ttft_p99_ms"),
        "inter_token_latency":     ("itl_avg_ms",  "itl_p99_ms"),
        "output_token_throughput": ("output_tps",  None),
        "request_throughput":      ("request_tps", None),
    }

    metrics = {}
    for jkey, (avg_k, p99_k) in key_map.items():
        if jkey not in data:
            continue
        val = data[jkey]
        if isinstance(val, dict):
            if avg_k and "avg"  in val: metrics[avg_k] = float(val["avg"])
            if p99_k and "p99"  in val: metrics[p99_k] = float(val["p99"])
        elif isinstance(val, (int, float)):
            if avg_k: metrics[avg_k] = float(val)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
#  RUN ONE GENAI-PERF INSTANCE
# ─────────────────────────────────────────────────────────────────────────────

def run_genai_perf(
    genai_perf_bin: str,
    server_url: str,        # e.g. "localhost:8000"
    served_model: str,      # name the vLLM server exposes
    concurrency: int,
    output_tokens: int,
    input_tokens: int,
    num_prompts: int,
    tokenizer: str,         # HF model name for synthetic prompt generation
    artifact_dir: str,
    log_path: str,
) -> SweepRow:

    cmd = [
        genai_perf_bin, "profile",
        "--model",              served_model,
        "--service-kind",       "openai",
        "--endpoint-type",      "chat",
        "--url",                server_url,
        "--concurrency",        str(concurrency),
        "--output-tokens-mean", str(output_tokens),
        "--output-tokens-stddev", "0",
        "--input-tokens-mean",  str(input_tokens),
        "--input-tokens-stddev", "0",
        "--num-prompts",        str(num_prompts),
        "--tokenizer",          tokenizer,
        "--artifact-dir",       artifact_dir,
        "--profile-export-file", "profile_export.json",
    ]

    print(f"  Running genai-perf (concurrency={concurrency})...", end=" ", flush=True)
    t0 = time.perf_counter()

    with open(log_path, "w") as log_fh:
        result = subprocess.run(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            timeout=300,
        )

    elapsed = time.perf_counter() - t0

    with open(log_path) as fh:
        stdout = fh.read()

    # Try JSON artifact first (most reliable), fall back to stdout
    metrics = parse_from_artifact(artifact_dir) or parse_from_stdout(stdout)

    row = SweepRow(
        concurrency=concurrency,
        mode="",  # set by caller
        log_path=log_path,
        **{k: metrics.get(k, 0.0) for k in
           ["ttft_avg_ms", "ttft_p99_ms", "itl_avg_ms",
            "itl_p99_ms", "output_tps", "request_tps"]},
    )
    row.parsed_ok = bool(metrics.get("output_tps"))

    status = "ok" if row.parsed_ok else "parse failed — check log"
    print(f"done in {elapsed:.0f}s  [{status}]")

    if not row.parsed_ok:
        print(f"  [WARN] Could not parse metrics. Raw output: {log_path}")
        print(f"         Check genai-perf version: genai-perf --version")

    return row


# ─────────────────────────────────────────────────────────────────────────────
#  SERVER LIFECYCLE  (same pattern as bench_serving.py)
# ─────────────────────────────────────────────────────────────────────────────

def start_vllm(target_model, port, served_name, extra_args=None, tp=1,
               log_path=None, env=None):
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

    log_fh = open(log_path, "w") if log_path else subprocess.DEVNULL
    proc = subprocess.Popen(
        cmd, stdout=log_fh, stderr=subprocess.STDOUT,
        env=env or os.environ.copy(), preexec_fn=os.setsid,
    )
    return proc, log_fh


def wait_for_server(port, timeout=360):
    import requests as req
    url = f"http://localhost:{port}/health"
    start = time.time()
    sys.stdout.write("  Waiting for vLLM")
    sys.stdout.flush()
    while time.time() - start < timeout:
        try:
            if req.get(url, timeout=2).status_code == 200:
                print(f"  ready in {time.time()-start:.0f}s")
                return True
        except Exception:
            pass
        time.sleep(4)
        sys.stdout.write(".")
        sys.stdout.flush()
    print("  TIMED OUT")
    return False


def stop_vllm(proc, log_fh=None):
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
    time.sleep(3)


# ─────────────────────────────────────────────────────────────────────────────
#  SWEEP ONE SERVER
# ─────────────────────────────────────────────────────────────────────────────

def sweep_server(
    genai_perf_bin, server_url, served_model,
    mode, concurrencies, output_tokens, input_tokens,
    num_prompts, tokenizer, base_dir,
) -> list:
    rows = []
    for c in concurrencies:
        artifact_dir = os.path.join(base_dir, mode, f"concurrency_{c}")
        log_path     = os.path.join(base_dir, mode, f"concurrency_{c}.log")
        os.makedirs(artifact_dir, exist_ok=True)

        row = run_genai_perf(
            genai_perf_bin=genai_perf_bin,
            server_url=server_url,
            served_model=served_model,
            concurrency=c,
            output_tokens=output_tokens,
            input_tokens=input_tokens,
            num_prompts=num_prompts,
            tokenizer=tokenizer,
            artifact_dir=artifact_dir,
            log_path=log_path,
        )
        row.mode = mode
        rows.append(row)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
#  OUTPUT: CSV + TABLE + OPTIONAL CHART
# ─────────────────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "concurrency", "mode",
    "ttft_avg_ms", "ttft_p99_ms",
    "itl_avg_ms",  "itl_p99_ms",
    "output_tps",  "request_tps",
    "speedup_vs_baseline",
]


def write_csv(baseline_rows: list, spec_rows: list, path: str):
    """Write combined CSV with speedup column."""
    baseline_by_c = {r.concurrency: r for r in baseline_rows}

    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for br in baseline_rows:
            row = br.as_dict()
            row["speedup_vs_baseline"] = 1.0
            writer.writerow(row)

        for sr in spec_rows:
            br = baseline_by_c.get(sr.concurrency)
            speedup = (sr.output_tps / br.output_tps) if br and br.output_tps else 0.0
            row = sr.as_dict()
            row["speedup_vs_baseline"] = round(speedup, 3)
            writer.writerow(row)

    print(f"\n  CSV written to: {path}")
    print(f"  Load in pandas: pd.read_csv('{path}')")


def print_table(baseline_rows: list, spec_rows: list, K: int):
    baseline_by_c = {r.concurrency: r for r in baseline_rows}

    print()
    hr("═")
    print("RESULTS — Throughput speedup by concurrency")
    hr("═")

    col = "  {:>13}  {:>12}  {:>12}  {:>10}  {:>10}  {:>10}"
    print(col.format("Concurrency", "AR tok/s", "SD tok/s", "Speedup",
                     "AR ITL p99", "SD ITL p99"))
    hr()

    for sr in spec_rows:
        br = baseline_by_c.get(sr.concurrency)
        if not br:
            continue
        speedup = sr.output_tps / br.output_tps if br.output_tps else 0
        marker  = "▲" if speedup > 1.05 else ("▼" if speedup < 0.95 else "≈")
        print(col.format(
            sr.concurrency,
            f"{br.output_tps:.1f}",
            f"{sr.output_tps:.1f}",
            f"{marker} {speedup:.2f}×",
            f"{br.itl_p99_ms:.1f}ms",
            f"{sr.itl_p99_ms:.1f}ms",
        ))

    hr()

    print()
    hr()
    print("TTFT comparison (p99 latency)")
    hr()
    col2 = "  {:>13}  {:>14}  {:>14}  {:>12}"
    print(col2.format("Concurrency", "AR TTFT p99", "SD TTFT p99", "Reduction"))
    hr()
    for sr in spec_rows:
        br = baseline_by_c.get(sr.concurrency)
        if not br:
            continue
        reduction = (1 - sr.ttft_p99_ms / br.ttft_p99_ms) if br.ttft_p99_ms else 0
        print(col2.format(
            sr.concurrency,
            f"{br.ttft_p99_ms:.1f}ms",
            f"{sr.ttft_p99_ms:.1f}ms",
            f"{reduction*100:+.1f}%",
        ))
    hr()


def plot_results(baseline_rows: list, spec_rows: list, out_path: str):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("  matplotlib not installed — skipping chart. pip install matplotlib")
        return

    baseline_by_c = {r.concurrency: r for r in baseline_rows}
    concurrencies = [r.concurrency for r in spec_rows]
    speedups      = []
    ar_tps        = []
    sd_tps        = []

    for sr in spec_rows:
        br = baseline_by_c.get(sr.concurrency)
        ar_tps.append(br.output_tps if br else 0)
        sd_tps.append(sr.output_tps)
        speedups.append(sr.output_tps / br.output_tps if br and br.output_tps else 1.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Speculative Decoding vs Autoregressive — Concurrency Sweep", fontsize=13)

    # Throughput chart
    ax1.plot(concurrencies, ar_tps, "o-", label="Autoregressive", color="#4C72B0")
    ax1.plot(concurrencies, sd_tps, "s-", label=f"Speculative decoding", color="#DD8452")
    ax1.set_xlabel("Concurrent requests")
    ax1.set_ylabel("Output tokens / second")
    ax1.set_title("Throughput")
    ax1.legend()
    ax1.set_xscale("log", base=2)
    ax1.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax1.set_xticks(concurrencies)
    ax1.grid(True, alpha=0.3)

    # Speedup chart
    ax2.plot(concurrencies, speedups, "D-", color="#55A868", linewidth=2)
    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, label="baseline (1.0×)")
    ax2.set_xlabel("Concurrent requests")
    ax2.set_ylabel("Speedup (speculative / autoregressive)")
    ax2.set_title("Speedup vs Concurrency")
    ax2.set_xscale("log", base=2)
    ax2.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax2.set_xticks(concurrencies)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Chart saved to: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GenAI-Perf concurrency sweep: speculative vs autoregressive.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--target",            default="facebook/opt-1.3b")
    parser.add_argument("--draft",             default="facebook/opt-125m")
    parser.add_argument("--ngram",             action="store_true",
                        help="Use ngram speculative decoding (no draft model, any target)")
    parser.add_argument("--num-spec-tokens",   type=int, default=5, dest="K")
    parser.add_argument("--concurrencies",     type=int, nargs="+",
                        default=CONCURRENCIES_DEFAULT, metavar="N",
                        help=f"Concurrency levels to sweep (default: {CONCURRENCIES_DEFAULT})")
    parser.add_argument("--output-tokens",     type=int, default=200, dest="output_tokens")
    parser.add_argument("--input-tokens",      type=int, default=128, dest="input_tokens")
    parser.add_argument("--num-prompts",       type=int, default=50,  dest="num_prompts")
    parser.add_argument("--port",              type=int, default=8000)
    parser.add_argument("--tp",               type=int, default=1,
                        help="Tensor-parallel size for multi-GPU")
    parser.add_argument("--hf-token",          default=os.environ.get("HF_TOKEN"), dest="hf_token")
    parser.add_argument("--out-dir",           default="sweep_results", dest="out_dir")
    parser.add_argument("--plot",              action="store_true",
                        help="Generate matplotlib chart (pip install matplotlib)")
    # External server mode — skip server management
    parser.add_argument("--baseline-url",      default=None, dest="baseline_url",
                        help="URL of already-running baseline vLLM (e.g. localhost:8000)")
    parser.add_argument("--speculative-url",   default=None, dest="speculative_url",
                        help="URL of already-running speculative vLLM")
    args = parser.parse_args()

    genai_perf_bin = check_genai_perf()
    external       = bool(args.baseline_url)
    if not external:
        check_vllm()

    os.makedirs(args.out_dir, exist_ok=True)

    env = os.environ.copy()
    if args.hf_token:
        env["HF_TOKEN"]               = args.hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = args.hf_token

    spec_mode = "ngram" if args.ngram else f"draft-K{args.K}"

    print()
    hr("═")
    print("GENAI-PERF CONCURRENCY SWEEP")
    hr("═")
    print(f"  Target model       : {args.target}")
    print(f"  Speculative mode   : {'ngram' if args.ngram else args.draft}")
    print(f"  K (spec tokens)    : {args.K}")
    print(f"  Concurrencies      : {args.concurrencies}")
    print(f"  Output tokens      : {args.output_tokens}")
    print(f"  Input tokens       : {args.input_tokens}")
    print(f"  Prompts per run    : {args.num_prompts}")
    print(f"  Results dir        : {args.out_dir}/")

    baseline_rows = []
    spec_rows     = []

    if external:
        # ── EXTERNAL mode ─────────────────────────────────────────────────────
        print()
        hr()
        print("EXTERNAL mode — using already-running servers")
        hr()
        served_name = args.target.split("/")[-1]

        if args.baseline_url:
            print(f"\n[1/2] Baseline sweep → {args.baseline_url}")
            baseline_rows = sweep_server(
                genai_perf_bin, args.baseline_url, served_name,
                "baseline", args.concurrencies,
                args.output_tokens, args.input_tokens, args.num_prompts,
                args.target, args.out_dir,
            )

        if args.speculative_url:
            print(f"\n[2/2] Speculative sweep → {args.speculative_url}")
            spec_rows = sweep_server(
                genai_perf_bin, args.speculative_url, "speculative",
                "speculative", args.concurrencies,
                args.output_tokens, args.input_tokens, args.num_prompts,
                args.target, args.out_dir,
            )

    else:
        # ── MANAGED mode — start/stop vLLM servers ────────────────────────────

        # 1. BASELINE
        print()
        hr()
        print("[1/2] BASELINE — autoregressive (no speculative decoding)")
        hr()
        baseline_log = os.path.join(args.out_dir, "vllm_baseline.log")
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", args.target, "--port", str(args.port),
            "--served-model-name", "baseline",
            "--max-model-len", "2048",
            "--gpu-memory-utilization", "0.85",
            "--disable-log-requests",
        ]
        if args.tp > 1:
            cmd += ["--tensor-parallel-size", str(args.tp)]

        proc, log_fh = start_vllm(
            args.target, args.port, "baseline",
            tp=args.tp, log_path=baseline_log, env=env,
        )
        try:
            if wait_for_server(args.port):
                baseline_rows = sweep_server(
                    genai_perf_bin,
                    f"localhost:{args.port}",
                    "baseline",
                    "baseline",
                    args.concurrencies,
                    args.output_tokens, args.input_tokens, args.num_prompts,
                    args.target, args.out_dir,
                )
            else:
                print(f"  Baseline server failed. See {baseline_log}")
        finally:
            stop_vllm(proc, log_fh)

        # 2. SPECULATIVE
        print()
        hr()
        print(f"[2/2] SPECULATIVE — {spec_mode}")
        hr()
        spec_log = os.path.join(args.out_dir, "vllm_speculative.log")

        if args.ngram:
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

        proc, log_fh = start_vllm(
            args.target, args.port, "speculative",
            extra_args=spec_extra, tp=args.tp,
            log_path=spec_log, env=env,
        )
        try:
            if wait_for_server(args.port):
                spec_rows = sweep_server(
                    genai_perf_bin,
                    f"localhost:{args.port}",
                    "speculative",
                    "speculative",
                    args.concurrencies,
                    args.output_tokens, args.input_tokens, args.num_prompts,
                    args.target, args.out_dir,
                )
            else:
                print(f"  Speculative server failed. See {spec_log}")
        finally:
            stop_vllm(proc, log_fh)

    # ── OUTPUT ────────────────────────────────────────────────────────────────
    if baseline_rows and spec_rows:
        print_table(baseline_rows, spec_rows, args.K)

        csv_path = os.path.join(args.out_dir, "sweep_results.csv")
        write_csv(baseline_rows, spec_rows, csv_path)

        if args.plot:
            chart_path = os.path.join(args.out_dir, "speedup_curve.png")
            plot_results(baseline_rows, spec_rows, chart_path)

        print()
        hr("═")
        print("INTERPRETING THE RESULTS")
        hr()
        print("""
  Speedup > 1.5× at concurrency=1   Good draft model / predictable workload.
  Speedup drops off quickly          GPU saturates early — consider larger GPU
                                     or reducing K (--num-spec-tokens).
  Speedup ≈ 1.0 at all concurrencies  Check: are draft+target models similar
                                     enough? Try a different draft model.
  TTFT improves but throughput doesn't  Typical at high concurrency — spec
                                     decoding helps latency but the GPU batch
                                     is already full.

  Load the CSV into a notebook for further analysis:

    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv("sweep_results/sweep_results.csv")
    pivot = df.pivot(index="concurrency", columns="mode", values="output_tps")
    pivot["speedup"] = pivot["speculative"] / pivot["baseline"]
    pivot["speedup"].plot(marker="o", title="Speedup vs Concurrency")
    plt.axhline(1.0, linestyle="--", color="gray")
    plt.show()
""")

    elif baseline_rows:
        print()
        print("[WARN] Only baseline results available (speculative server failed).")
        csv_path = os.path.join(args.out_dir, "baseline_results.csv")
        write_csv(baseline_rows, [], csv_path)

    print()
    hr("═")


if __name__ == "__main__":
    main()
