"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture

import numpy as np

import numpy as np

def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a Gaussian component.
    
    Args:
        X: (n, d) array holding the data.
        mixture: the current Gaussian mixture (means, variances, and weights).
    
    Returns:
        np.ndarray: (n, K) array holding the soft counts (responsibilities) for all components for all examples.
        float: log-likelihood of the assignment.
    """
    n, d = X.shape
    K = mixture.mu.shape[0]
    
    # Initialize the responsibilities matrix
    post = np.zeros((n, K))
    p = np.zeros((n, K))
    # Calculate responsibilities for each data point and each Gaussian component
    for i in range(n):
        for j in range(K):
            diff = X[i] - mixture.mu[j]
            # Covariance matrix (diagonal matrix)
            sigma = np.diag([mixture.var[j]] * d)
            # Compute the exponent term
            inv_sigma = np.linalg.inv(sigma)
            exponent = -0.5 * np.dot(diff.T, np.dot(inv_sigma, diff))
            # Compute the normalization coefficient
            coef = 1 / (np.sqrt((2 * np.pi)**d * np.linalg.det(sigma)))
            # Compute the responsibility
            post[i, j] = mixture.p[j] * coef * np.exp(exponent)
        
        # Normalize the responsibilities so they sum to 1 for each data point
        p[i,:] = post[i, :]
        post[i, :] /= np.sum(post[i, :])
    
    # Compute the log-likelihood
    log_likelihood = np.sum(np.log(np.sum(p, axis=1)))
    
    return post, log_likelihood
    
   

def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape

    n_hat = post.sum(axis=0)
    p = n_hat / n

    mu = np.zeros((K, d))
    var = np.zeros(K)

    for j in range(K):
        mu[j, :] = post[:, j] @ X / n_hat[j]
        sse = ((mu[j] - X)**2).sum(axis=1) @ post[:, j]
        var[j] = sse / (d * n_hat[j])

    return GaussianMixture(mu, var, p)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    prev_cost = None
    cost = None
    while (prev_cost is None or cost - prev_cost > (1e-6)*abs(cost)):
        prev_cost = cost
        post, cost = estep(X, mixture)
        mixture = mstep(X, post, mixture)

    return mixture, post, cost



def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError
