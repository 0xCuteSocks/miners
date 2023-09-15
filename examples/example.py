import numpy as np
import miners
from time import time


def print_stats(score):
    print("MIC", miners.mine_mic(score))
    print("MAS", miners.mine_mas(score))
    print("MEV", miners.mine_mev(score))
    print("MCN (eps=0)", miners.mine_mcn(score, 0.0))
    print("MCN (eps=1-MIC)", miners.mine_mcn_general(score))
    print("TIC", miners.mine_tic(score, False))


x = np.linspace(0, 100000, 100000)
y = np.sin(10 * np.pi * x) + x
t = time()
param = miners.MineParameter(alpha=0.6, c=15)
prob = miners.MineProblem(x, y, param)
score = miners.mine_compute_score(prob, param)
end = time() - t
print("Without noise:")
print_stats(score)
print(f"\ntime cost: {end}s")

np.random.seed(0)
y += np.random.uniform(0, 50000, x.shape[0])  # add some noise
t = time()
prob = miners.MineProblem(x, y, param)
score = miners.mine_compute_score(prob, param)
end = time() - t
print("With noise:")
print_stats(score)
print(f"\ntime cost: {end}s")
