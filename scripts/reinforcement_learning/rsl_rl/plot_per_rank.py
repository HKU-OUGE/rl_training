#!/usr/bin/env python3
"""Quick visualizer for /tmp/per_rank_train_*_rank*.jsonl files.

Reads each rank's jsonl (locally or via scp), plots terrain_level_mean
and terrain_level_max over training iterations, one subplot per metric,
one line per rank.

Usage:
    # Pull data from 102 and plot:
    python plot_per_rank.py --remote 102_container

    # Or plot from local files (already pulled):
    python plot_per_rank.py --local /tmp

    # Save to file instead of showing:
    python plot_per_rank.py --remote 102_container --save /tmp/per_rank.png

Default: pulls from 102_container and shows interactive plot.
"""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def fetch_remote_jsonls(remote: str, dest: Path) -> Path:
    """SCP all /tmp/per_rank_train_*_rank*.jsonl from remote to local dest dir."""
    dest.mkdir(parents=True, exist_ok=True)
    cmd = f"scp -q '{remote}:/tmp/per_rank_train_*_rank*.jsonl' {shlex.quote(str(dest))}/"
    print(f"[fetch] {cmd}")
    subprocess.run(cmd, shell=True, check=False)
    return dest


METRICS = [
    ("terrain_level_mean", "terrain_level_mean"),
    ("terrain_level_max",  "terrain_level_max"),
    ("value_function",     "PPO value loss (per-rank critic MSE)"),
    ("surrogate",          "PPO surrogate loss"),
    ("entropy",            "policy entropy"),
    ("ep_reward",          "episode total reward (rewbuffer mean)"),
    ("ep_length",          "episode length (lenbuffer mean)"),
    ("term_time_out",                 "termination: time_out rate"),
    ("term_illegal_contact",          "termination: illegal_contact rate"),
    ("term_terrain_out_of_bounds",    "termination: terrain_out_of_bounds rate"),
]


def load_rank_data(local_dir: Path):
    """Return {rank_name: [dict_row, ...]}. Each row keeps all fields from jsonl."""
    data = {}
    for f in sorted(local_dir.glob("per_rank_train_*_rank*.jsonl")):
        stem = f.stem  # per_rank_train_NAME_rankN
        try:
            name_part = stem[len("per_rank_train_"):].rsplit("_rank", 1)[0]
            rank_id_str = stem.rsplit("_rank", 1)[1]
            rank_id = int(rank_id_str)
        except Exception:
            print(f"[skip] {f.name}: cannot parse rank from filename")
            continue
        rows = []
        with open(f) as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if rows:
            key = f"rank{rank_id}_{name_part}"
            data[key] = rows
            latest = rows[-1]
            extras = ""
            if "value_function" in latest:
                extras = f", vloss={latest['value_function']:.3f}"
            print(f"[load] {key}: {len(rows)} rows, iter {rows[0].get('iter',0)}-"
                  f"{rows[-1].get('iter',0)}, level_mean={latest.get('terrain_level_mean', float('nan')):.2f}{extras}")
    return data


def plot(data, save: Path | None = None):
    if not data:
        print("No data to plot — did the SCP succeed?")
        sys.exit(1)
    # Detect which metrics are present in the latest row across all ranks
    available = []
    for key, label in METRICS:
        if any(key in r for rows in data.values() for r in rows):
            available.append((key, label))
    if not available:
        print("No known metric columns found.")
        sys.exit(1)
    fig, axes = plt.subplots(len(available), 1,
                             figsize=(11, 3 * len(available)),
                             sharex=True, squeeze=False)
    ordered = sorted(data.items(), key=lambda kv: int(kv[0].split("_", 1)[0][4:]))
    for ax_idx, (key, label) in enumerate(available):
        ax = axes[ax_idx, 0]
        for rank_key, rows in ordered:
            xs, ys = [], []
            for r in rows:
                if key in r:
                    xs.append(r.get("iter", 0))
                    ys.append(r[key])
            if xs:
                ax.plot(xs, ys, label=rank_key, lw=1.3)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.3)
        if ax_idx == 0:
            ax.set_title("Per-rank training metrics")
        if ax_idx == len(available) - 1:
            ax.set_xlabel("iter")
        ax.legend(loc="upper left", fontsize=7, ncol=2)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=120)
        print(f"[save] {save}")
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--remote", help="SSH host (e.g. 102_container) to pull jsonl from")
    g.add_argument("--local", type=Path, help="Local dir containing jsonl files")
    p.add_argument("--save", type=Path, help="Save to PNG instead of interactive show")
    args = p.parse_args()

    if args.remote:
        scratch = Path("/tmp/per_rank_pull")
        fetch_remote_jsonls(args.remote, scratch)
        local_dir = scratch
    elif args.local:
        local_dir = args.local
    else:
        # Default: pull from 102_container
        scratch = Path("/tmp/per_rank_pull")
        fetch_remote_jsonls("102_container", scratch)
        local_dir = scratch

    data = load_rank_data(local_dir)
    plot(data, save=args.save)


if __name__ == "__main__":
    main()
