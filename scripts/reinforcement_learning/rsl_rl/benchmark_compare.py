"""Compare benchmark_scaling.py outputs and print scaling summary.

Usage:
    python benchmark_compare.py /tmp/bench_*.json
"""

import json
import sys


def load(path):
    with open(path) as f:
        return json.load(f)


def main(paths):
    reports = sorted([load(p) for p in paths], key=lambda r: r["world_size"])
    if not reports:
        print("no reports")
        return

    base = reports[0]
    base_iter = base["steady_iter_mean"]
    base_thru = base["throughput_steps_per_sec"]

    print()
    print(f"{'GPUs':>5s} | {'env/gpu':>7s} | {'total env':>9s} | {'iter mean':>9s} | "
          f"{'iter p50':>8s} | {'throughput':>13s} | {'iter speedup':>12s} | {'thru speedup':>12s}")
    print("-" * 110)
    for r in reports:
        ws = r["world_size"]
        ne = r["num_envs_per_gpu"]
        tot = ne * ws
        m = r["steady_iter_mean"]
        p50 = r.get("steady_iter_p50") or m
        thru = r["throughput_steps_per_sec"]
        iter_sp = base_iter / m if m > 0 else float("inf")
        thru_sp = thru / base_thru if base_thru > 0 else float("nan")
        print(f"{ws:>5d} | {ne:>7d} | {tot:>9d} | {m:>8.3f}s | {p50:>7.3f}s | "
              f"{thru:>10.0f} st/s | {iter_sp:>11.2f}x | {thru_sp:>11.2f}x")
    print()

    # interpretation hint
    print("Reading guide:")
    print("  - 'iter speedup'  = how much faster ONE iteration runs vs 1-GPU baseline.")
    print("    For weak scaling (env/gpu fixed) this stays ~1.0x — that's expected.")
    print("  - 'thru speedup'  = how much MORE total experience is collected per second.")
    print("    Scales with world_size if comm overhead is small. Want close to N-x for N GPUs.")
    print("  - If throughput speedup << N-x, you have DDP overhead or sim sync stragglers.")

    # warmup commentary
    print()
    print("Warmup (iter 0) costs:")
    for r in reports:
        print(f"  ws={r['world_size']:>2d}: warmup={r['warmup_wall_time']:.1f}s  "
              f"first 5 iter mean={sum(r['iter_wall_times'][:5]) / 5.0:.3f}s")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: benchmark_compare.py <bench_1gpu.json> [<bench_4gpu.json> ...]")
        sys.exit(1)
    main(sys.argv[1:])
