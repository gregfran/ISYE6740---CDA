import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat
from sklearn.decomposition import PCA

def load_data():
    # get data/imgs mat
    datamat = loadmat('data/data.mat')
    # get labels mat
    labelmat = loadmat('data/label.mat')

    # get data and labels from dicts
    data = datamat['data']
    labels = labelmat['trueLabel']

    return data, labels

def em_gmm(data, comps, i_max, tol):
    np.random.seed(19) # joe sakic
    
    # data is n (samples) x d (features)
    n, d = data.shape
    c = comps # C=2 per assignment

    # param init
        # mean init
    mu = np.random.randn(c, d) # c x d, rows are mean vecs
        # cov init
    sigma = np.array([np.eye(d) for _ in range(c)]) # c x d x d
        # pi init
    pi = np.ones(c) / c # normalized to sum to 1
        # log likelihood init
    log_liks = []

    for i in range(i_max):
        # e-step expectations
        # init array for posteriors
        tau = np.zeros((n, c)) # n x c, rows are samples, cols are comps
        for k in range(c):
            # likelihood of each sample under comp k
            # delta term, data == x_i
            deltas = data - mu[k]
            # compute likelihood using multivariate normal pdf
            inv = np.linalg.inv(sigma[k]) # inverse of cov
            det = np.linalg.det(sigma[k]) # determinant of cov
            # p(x_i | mu_k, sigma_k) = 1/sqrt((2pi)^d * det) * exp(-0.5 * (x_i - mu_k)^T * inv * (x_i - mu_k))
            # normalization constant
            p_x = 1.0 / np.sqrt((2 * np.pi) ** d * det)
            # exponent term
            exp_term = np.exp(-0.5 * np.sum(deltas @ inv * deltas, axis=1))



def main():
    # get dir correct
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    # results folder
    os.makedirs('results', exist_ok=True)
    # get the data from .mat
    data, labels = load_data()
    # data is 784 x 1990 so I need to transpose it to get 1990 x 784 for sklearn
    # labels is 1 x 1990 so I should flatten to get 1990
    dataT = data.T # 1990 x 784
    labelsF = labels.flatten() # 1990

    # assignment requires PCA w/ 4 comps to reduce dimensionality before doing EM
    pca = PCA(n_components=4)
    dataT_pca = pca.fit_transform(dataT) # 1990 x 4
    print(dataT_pca.shape)

    # EM GMM implementation




    


    pass

if __name__ == "__main__":
    main()