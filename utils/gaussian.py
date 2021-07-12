import numpy as np

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
