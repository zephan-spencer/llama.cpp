#!/usr/bin/env python3

"""Parse llama-bench MoE-cache JSONL and make a cache-size line plot.

Examples:
  scripts/plot-moe-cache.py benchmark-results/qwen-single.jsonl \
      --label single --plot benchmark-results/qwen-cache.png

  scripts/plot-moe-cache.py \
      benchmark-results/qwen-single.jsonl benchmark-results/qwen-tensor.jsonl \
      --label single --label tensor \
      --csv benchmark-results/qwen-cache.csv \
      --plot benchmark-results/qwen-cache.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


TEST_ORDER = ("prompt", "generation", "prompt+generation")


def test_name(record: dict[str, Any]) -> str:
    has_prompt = int(record.get("n_prompt", 0)) > 0
    has_generation = int(record.get("n_gen", 0)) > 0
    if has_prompt and has_generation:
        return "prompt+generation"
    if has_prompt:
        return "prompt"
    if has_generation:
        return "generation"
    return "unknown"


def read_records(path: Path, label: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                continue
            if "avg_ts" not in record or "n_moe_cache_experts" not in record:
                continue

            mode = test_name(record)
            if mode == "unknown":
                continue
            cache_experts = int(record["n_moe_cache_experts"])
            if cache_experts < 0:
                raise ValueError(f"{path}:{line_number}: negative cache size")
            avg_ts = float(record["avg_ts"])
            stddev_ts = float(record.get("stddev_ts", 0.0))
            if not math.isfinite(avg_ts) or not math.isfinite(stddev_ts):
                raise ValueError(f"{path}:{line_number}: non-finite throughput")

            records.append(
                {
                    "run": label,
                    "test": mode,
                    "cache_experts": cache_experts,
                    "cache_label": "fully resident" if cache_experts == 0 else str(cache_experts),
                    "avg_ts": avg_ts,
                    "stddev_ts": stddev_ts,
                    "n_prompt": int(record.get("n_prompt", 0)),
                    "n_gen": int(record.get("n_gen", 0)),
                    "model_filename": record.get("model_filename", ""),
                    "model_type": record.get("model_type", ""),
                    "devices": record.get("devices", ""),
                    "split_mode": record.get("split_mode", ""),
                    "tensor_split": record.get("tensor_split", ""),
                    "source": str(path),
                }
            )
    return records


def cache_order(records: list[dict[str, Any]]) -> list[int]:
    values = {int(record["cache_experts"]) for record in records}
    return ([0] if 0 in values else []) + sorted(value for value in values if value != 0)


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run",
        "test",
        "cache_experts",
        "cache_label",
        "avg_ts",
        "stddev_ts",
        "n_prompt",
        "n_gen",
        "model_filename",
        "model_type",
        "devices",
        "split_mode",
        "tensor_split",
        "source",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def make_plot(
    path: Path,
    records: list[dict[str, Any]],
    selected_tests: set[str],
    title: str,
    no_errorbars: bool,
    dpi: int,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to create the plot") from exc

    records = [record for record in records if record["test"] in selected_tests]
    if not records:
        raise ValueError("no records match the selected test type")

    order = cache_order(records)
    positions = {cache: index for index, cache in enumerate(order)}
    labels = {0: "fully resident", **{cache: str(cache) for cache in order if cache != 0}}
    run_names = list(dict.fromkeys(str(record["run"]) for record in records))
    tests = [
        test
        for test in TEST_ORDER
        if test in selected_tests and any(record["test"] == test for record in records)
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.get_cmap("tab10")
    line_styles = {"prompt": "-", "generation": "--", "prompt+generation": ":"}

    for run_index, run in enumerate(run_names):
        color = colors(run_index % 10)
        for test in tests:
            points = {
                int(record["cache_experts"]): record
                for record in records
                if record["run"] == run and record["test"] == test
            }
            if not points:
                continue
            x = [positions[cache] for cache in order if cache in points]
            y = [float(points[cache]["avg_ts"]) for cache in order if cache in points]
            yerr = [float(points[cache]["stddev_ts"]) for cache in order if cache in points]
            kwargs = {
                "color": color,
                "linestyle": line_styles[test],
                "marker": "o",
                "linewidth": 2,
                "capsize": 3,
                "label": f"{run} / {test}",
            }
            if no_errorbars:
                ax.plot(x, y, **kwargs)
            else:
                ax.errorbar(x, y, yerr=yerr, **kwargs)

    ax.set_xticks(range(len(order)), [labels[cache] for cache in order])
    ax.set_xlabel("Cached logical experts per MoE layer")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="llama-bench JSONL output files")
    parser.add_argument(
        "--label",
        action="append",
        help="label for an input file; repeat once per input (default: input filename)",
    )
    parser.add_argument("--csv", type=Path, help="parsed CSV output")
    parser.add_argument("--plot", type=Path, help="line plot output (PNG, SVG, or another matplotlib format)")
    parser.add_argument(
        "--test",
        choices=("prompt", "generation", "both"),
        default="both",
        help="which benchmark type to plot (default: both)",
    )
    parser.add_argument("--title", default="MoE expert-cache benchmark")
    parser.add_argument("--no-errorbars", action="store_true", help="omit llama-bench standard-deviation error bars")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.label is not None and len(args.label) != len(args.inputs):
        raise SystemExit("error: --label must be supplied once for each input file")
    labels = args.label or [path.stem for path in args.inputs]

    records: list[dict[str, Any]] = []
    for path, label in zip(args.inputs, labels):
        if not path.is_file():
            raise SystemExit(f"error: input does not exist: {path}")
        records.extend(read_records(path, label))
    if not records:
        raise SystemExit("error: no llama-bench records found")

    csv_path = args.csv
    if csv_path is None:
        csv_path = args.inputs[0].with_suffix(".csv") if len(args.inputs) == 1 else Path("moe-cache-comparison.csv")
    plot_path = args.plot
    if plot_path is None:
        plot_path = args.inputs[0].with_suffix(".png") if len(args.inputs) == 1 else Path("moe-cache-comparison.png")

    write_csv(csv_path, records)
    selected_tests = set(TEST_ORDER if args.test == "both" else (args.test,))
    try:
        make_plot(plot_path, records, selected_tests, args.title, args.no_errorbars, args.dpi)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    print(f"Wrote {len(records)} records to {csv_path}")
    print(f"Wrote line plot to {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
