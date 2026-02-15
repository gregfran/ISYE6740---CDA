import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as kde

def load_data():
    data = pd.read_csv('data/n90pol.csv', header=True)
    return data

def main():
    # ----- QUESTION 1 -----
    # load n90pol dataset
    data = load_data()

    # process data
    amyg = data['amygdala'].values
    acc = data['acc'].values


    pass

if __name__ == "__main__":
    main()