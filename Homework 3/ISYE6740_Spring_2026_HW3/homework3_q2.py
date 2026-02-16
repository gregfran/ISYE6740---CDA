import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as kde
import os

def load_data():
    data = pd.read_csv('data/n90pol.csv')
    return data

def histogrammer(data, bins, name, xlabel, ylabel, ax):
    ax.hist(data, bins=bins, density=True, alpha=0.5, color='blue', edgecolor='black')
    ax.set_title(f'{name} Histogram {bins} bins', fontsize=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(alpha=0.2)

def kde_plotter(data, name, xlabel, ylabel, ax, factor_adj):
    data_kde = kde(data)
    # smaller factor is less smooth; >1 for more smooth
    data_kde.set_bandwidth(data_kde.factor * factor_adj)

    # create eval points, evenely spaced between min and max of data
    data_range = data.max() - data.min()
    data_pts = np.linspace(data.min(), data.max(), 400)
    data_kde_vals = data_kde(data_pts)

    # plot KDE
    ax.plot(data_pts, data_kde_vals, color='blue', label=f'{name} KDE')
    ax.set_title(f'{name} KDE {factor_adj} factor', fontsize=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(alpha=0.2)
    ax.fill_between(data_pts, data_kde_vals, alpha=0.5, color='blue')


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    # results folder
    os.makedirs('results', exist_ok=True)

    # ----- QUESTION 1 -----
    # load n90pol dataset
    data = load_data()

    # process data
    amyg = data['amygdala'].values
    acc = data['acc'].values

    # subplots for amyg plots
    fig, axs = plt.subplots(2, 1, figsize=(15,10))

    # amyg histogram
    amyg_bins = 24
    histogrammer(amyg, amyg_bins, 'Amygdala', 'Amygdala', 'Density', axs[0])
    # amyg KDE
    kde_plotter(amyg, 'Amygdala', 'Amygdala', 'Density', axs[1], 0.5)

    plt.tight_layout()
    plt.savefig('results/amygdala_plots.png')

    # subplots for acc plots
    fig2, axs2 = plt.subplots(2, 1, figsize=(15,10))
    # acc histogram
    acc_bins = 24
    histogrammer(acc, acc_bins, 'ACC', 'ACC', 'Density', axs2[0])
    # acc KDE
    kde_plotter(acc, 'ACC', 'ACC', 'Density', axs2[1], 0.5)

    plt.tight_layout()
    plt.savefig('results/acc_plots.png')

    pass

if __name__ == "__main__":
    main()