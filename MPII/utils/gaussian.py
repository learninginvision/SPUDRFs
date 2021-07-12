import numpy as np
import scipy.stats as st
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

def gaussian_func(y, mu, sigma):
    samples = y.shape[0]
    num_tree, leaf_num, _, _ = mu.shape
    
    y = np.reshape(y, [samples, 1, 1])
    y = np.repeat(y, num_tree, 1)
    y = np.repeat(y, leaf_num, 2)   

    mu = np.reshape(mu, [1, num_tree, leaf_num])
    mu = mu.repeat(samples, 0)

    sigma = np.reshape(sigma, [1, num_tree, leaf_num])
    sigma = sigma.repeat(samples, 0)  

    res = 1.0 / np.sqrt(2 * 3.14 * (sigma + 1e-9)) * \
         (np.exp(- (y - mu) ** 2 / (2 * (sigma + 1e-9))) + 1e-9)

    return res

def multi_gaussian(y, mu, sigma):
    '''
    y has the shape of [samples, task_num]
    mu has the shape of [num_tree, leaf_node_per_tree, task_num, 1]
    sigma has the shape of [num_tree, leaf_node_per_tree, task_num, task_num]
    '''
    samples = y.shape[0]
    num_tree, leaf_num, task_num, _ = mu.shape
    gauss_val = np.zeros((samples, num_tree, leaf_num))
    
    mu = mu.reshape(num_tree, leaf_num, task_num)
    
    for i in range(num_tree):
        for j in range(leaf_num):
            t = st.multivariate_normal.pdf(y, mean=mu[i, j, :], cov=sigma[i, j, :, :], allow_singular=True)
            gauss_val[:, i, j] = t
    return gauss_val

def multi_gaussian_torch(y, mu, sigma):
    '''
    y has the shape of [samples, task_num]
    mu has the shape of [num_tree, leaf_node_per_tree, task_num, 1]
    sigma has the shape of [num_tree, leaf_node_per_tree, task_num, task_num]
    '''
    samples = y.shape[0]
    num_tree, leaf_num, task_num, _ = mu.shape
    gauss_val = np.zeros((samples, num_tree, leaf_num))
    mu = mu.reshape(num_tree, leaf_num, task_num)
    
    for i in range(num_tree):
        for j in range(leaf_num):
            t = MultivariateNormal(mu[i, j, :], sigma[i, j, :, :]).log_prob(y)
            gauss_val[:, i, j] = torch.exp(t)
    return gauss_val
