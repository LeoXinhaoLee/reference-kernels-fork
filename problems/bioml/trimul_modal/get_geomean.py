#!/usr/bin/env python3
import math
import re
import sys
from pathlib import Path
from typing import List, Tuple

MEAN_RE = re.compile(r"benchmark\.\d+\.mean:\s*([0-9.eE+-]+)")
TEST_RE = re.compile(r"(?:^|/|\\)test(\d+)(?:/|\\|$)")

# Common device encodings in logs
DEVICE_RES = [
    re.compile(r"^\s*device\s*[:=]\s*(.+?)\s*$", re.IGNORECASE),
    re.compile(r"^\s*Device\s*[:=]\s*(.+?)\s*$", re.IGNORECASE),
    re.compile(r"^\s*GPU\s*[:=]\s*(.+?)\s*$", re.IGNORECASE),
    re.compile(r"^\s*NVIDIA\s+(.+)$", re.IGNORECASE),
]


def parse_test_i(path: str) -> str:
    m = TEST_RE.search(path)
    return f"test{m.group(1)}" if m else "unknown"


def parse_device(lines: List[str], path: str) -> str:
    for line in lines:
        for rx in DEVICE_RES:
            m = rx.search(line)
            if m:
                return m.group(1).strip()

    # fallback: infer from path (e.g. .../A100/...)
    for part in Path(path).parts:
        if re.fullmatch(r"[A-Za-z]\w{1,20}", part) and not re.fullmatch(r"test\d+", part, re.IGNORECASE):
            return part

    return "unknown"


def geometric_stats(values: List[float]) -> Tuple[float, float]:
    logs = [math.log(v) for v in values]
    log_mean = sum(logs) / len(logs)
    # geom_mean = math.exp(log_mean)
    geom_mean = math.pow(math.prod(values), 1.0 / len(values))  # https://github.com/gpu-mode/discord-cluster-manager/blob/c3a50838e982f34fceb329cf1d77a29d503333f2/src/libkernelbot/submission.py#L193

    log_var = sum((l - log_mean) ** 2 for l in logs) / len(logs)
    geom_std = math.exp(math.sqrt(log_var))

    return geom_mean, geom_std


def process_file(filename: str) -> Tuple[str, str, float, float]:
    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    means = []
    for line in lines:
        m = MEAN_RE.search(line)
        if m:
            means.append(float(m.group(1)))

    if not means:
        raise ValueError(f"No benchmark.*.mean values found in {filename}")

    device = parse_device(lines, filename)
    test_i = parse_test_i(filename)

    geom_mean_ns, geom_std = geometric_stats(means)
    geom_mean_us = geom_mean_ns / 1_000.0  # ns → µs

    return device, test_i, geom_mean_us, geom_std


def main() -> int:
    out_path = 'results_all.tsv'
    input_files = [
        f'./out_best/A100-80GB/test{i}' for i in range(1, 7)
    ]
    input_files += [
        f'./out_best/H100/test{i}' for i in range(1, 7)
    ]
    input_files += [
        f'./out_best/B200/test{i}' for i in range(1, 4)
    ]

    if not input_files:
        print("No input files provided", file=sys.stderr)
        return 2

    with open(out_path, "w", encoding="utf-8") as out:
        out.write("device\ttest_i\tgeometric_mean\tgeometric_std\n")

        for fn in input_files:
            device, test_i, gmean, gstd = process_file(fn + '/leaderboard.out')
            out.write(f"{device}\t{test_i}\t{gmean:.6f}\t{gstd:.6f}\n")

    return 0


if __name__ == "__main__":
    main()
