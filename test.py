import numpy as np
import em
import common

X = np.loadtxt("test_incomplete.txt")
X_gold = np.loadtxt("test_complete.txt")

K = 4
n, d = X.shape
seed = 0
X = np.loadtxt("toy_data.txt")

K = [1,2,3,4]
Seed = [0,1,2,3,4]

for k in K:
    for seed in Seed:
        mixture, ndaaray = common.init(X, k, seed)
        n, _ = X.shape
        post = np.zeros((n,k))
        mixture, post, cost = em.run(X,mixture,post)
        bic = common.bic(X,mixture, cost)
        print(k,seed, cost, bic)
