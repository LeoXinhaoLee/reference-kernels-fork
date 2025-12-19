import math
import re
import sys


filename = "./out_best/test3/leaderboard.out"

means = []

pattern = re.compile(r"benchmark\.\d+\.mean:\s*([0-9.eE+-]+)")

with open(filename, "r") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            means.append(float(m.group(1)))

if not means:
    print("No benchmark.*.mean values found")
    sys.exit(1)

# geometric mean = exp(mean(log(x)))
log_sum = sum(math.log(x) for x in means)
geom_mean = math.exp(log_sum / len(means))

# convert ns → us
geom_mean_us = geom_mean / 1_000.0

print(f"Geometric mean of benchmark means: {geom_mean_us:.3f} us")
