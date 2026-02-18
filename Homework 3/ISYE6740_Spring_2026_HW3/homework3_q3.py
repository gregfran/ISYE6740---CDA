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
    dataT_pca = pca.fit_transform(dataT)
    print(dataT_pca.shape)


    pass

if __name__ == "__main__":
    main()