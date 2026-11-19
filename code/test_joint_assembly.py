"""Known-answer checks for the T23 joint-metric assembly. Run directly; prints PASS/FAIL."""

import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heads

heads.BASE = str(Path(__file__).resolve().parent.parent)

from joint_assembly import _joint_for, _population

names = [f"L{i}" for i in range(5)]
rng = np.random.default_rng(0)
n = 13448

y = rng.integers(0, 2, (n, 5)).astype(float)
y[rng.random((n, 5)) < 0.05] = -1.0

complete, y_mat = _population(y, names, names)
print(f"_population: {complete.sum()}/{n} complete rows, y_mat={y_mat.shape}")
assert not (y_mat == -1).any(), "uncertain labels leaked into the evaluation matrix"

thr = np.full((5, 5, 2), 0.5, dtype=np.float32)

perfect = np.zeros((5, 5, 2, n), dtype=np.float32)
for li in range(5):
    perfect[li, :, :] = np.where(y[:, li] == 1, 0.9, 0.1)

m = _joint_for(perfect, thr, y_mat, complete, [0] * 5)
print("perfect  ->", {k: round(v["mean"], 4) for k, v in m.items()})
assert m["hamming_loss"]["mean"] == 0.0
assert m["exact_match"]["mean"] == 1.0

m2 = _joint_for(1 - perfect, thr, y_mat, complete, [0] * 5)
print("inverted ->", {k: round(v["mean"], 4) for k, v in m2.items()})
assert m2["hamming_loss"]["mean"] == 1.0
assert m2["exact_match"]["mean"] == 0.0

random_probs = rng.random((5, 5, 2, n)).astype(np.float32)
start = time.time()
count = 0
for combo in itertools.product(range(5), repeat=5):
    _joint_for(random_probs, thr, y_mat, complete, list(combo))
    count += 1
elapsed = time.time() - start
print(
    f"assembly : {count} combinations at 2 seeds in {elapsed:.1f}s "
    f"(projected {elapsed * 2.5:.0f}s at 5 seeds)"
)

print("ALL CHECKS PASSED")
