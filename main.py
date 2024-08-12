import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

K = [1,2,3,4]
Seed = [0,1,2,3,4]
for k in K:
    for seed in Seed:
        mixture, ndaaray = common.init(X, k, seed)
        n, _ = X.shape
        post = np.random.rand(n, k)
        post = post / post.sum(axis=1, keepdims=True)
        mixture, post, cost =kmeans.run(X,mixture,post)
        print(k, seed, cost)
        # TODO: Your code here
