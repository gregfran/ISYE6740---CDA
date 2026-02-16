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
    data_pts = np.linspace(data.min(), data.max(), 400)
    data_kde_vals = data_kde(data_pts)

    # plot KDE
    ax.plot(data_pts, data_kde_vals, color='blue', label=f'{name} KDE')
    ax.set_title(f'{name} KDE {factor_adj} factor', fontsize=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(alpha=0.2)
    ax.fill_between(data_pts, data_kde_vals, alpha=0.5, color='blue')

def kde2d_plotter(data_x, data_y, name_x, name_y, ax, factor_adj):
    # 2d kde
    kde_2d = kde([data_x, data_y])
    # bw to factor_adj
    kde_2d.set_bandwidth(kde_2d.factor * factor_adj)

    # create eval points, same way as before but for both datasets
    x_pts = np.linspace(data_x.min(), data_x.max(), 400)
    y_pts = np.linspace(data_y.min(), data_y.max(), 400)
    # create grid of eval points
    x_grid, y_grid = np.meshgrid(x_pts, y_pts)

    # evaluate kde on grid
    coords = np.vstack([x_grid.ravel(), y_grid.ravel()])
    kde_2d_vals = kde_2d(coords)
    kde_2d_vals = kde_2d_vals.reshape(x_grid.shape) # return to grid shape

    # heatmap for simplicity
    hm = ax.imshow(kde_2d_vals,
                   origin='lower',
                   aspect='auto',
                   extent=[data_x.min(), data_x.max(), data_y.min(), data_y.max()],
                   cmap='Blues'
                   )
    ax.set_xlabel(name_x, fontsize=10)
    ax.set_ylabel(name_y, fontsize=10)
    ax.set_title(f'{name_x} vs {name_y} 2D KDE {factor_adj} factor', fontsize=12)
    plt.colorbar(hm, ax=ax, label='Density')

def kde_cond_plotter(data, ori, name, ax, factor_adj):
    # get all unique orientations
    orients = data['orientation'].unique()
    sorted_orients = sorted(orients) # sort for plots later

    for idx, o in enumerate(sorted_orients):
        o_data = data[data['orientation'] == o][name].values

        # we need more than 2 pts to plot KDE
        if len(o_data) > 2:
            o_kde = kde(o_data)
            o_kde.set_bandwidth(o_kde.factor * factor_adj)

            # create eval points
            o_pts = np.linspace(o_data.min(), o_data.max(), 400)
            o_kde_vals = o_kde(o_pts)

            # plot KDE
            ax.plot(o_pts, o_kde_vals, label=f'{ori} {o}')
    ax.set_title(f'{name} KDE by {ori} {factor_adj} factor', fontsize=12)
    ax.set_xlabel(name, fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.grid(alpha=0.2)
    ax.legend()

def kde2d_cond_plotter(data, ori, name_x, name_y, ax, factor_adj):
    # get all unique orientations
    orients = data['orientation'].unique()
    sorted_orients = sorted(orients) # sort for plots later

    o_data_x = data[data['orientation'] == ori][name_x].values
    o_data_y = data[data['orientation'] == ori][name_y].values

    # we need more than 2 pts to plot KDE
    if len(o_data_x) > 2 and len(o_data_y) > 2:
        kde_2d = kde([o_data_x, o_data_y])
        kde_2d.set_bandwidth(kde_2d.factor * factor_adj)

        # create eval points, same way as before but for both datasets
        x_pts = np.linspace(o_data_x.min(), o_data_x.max(), 400)
        y_pts = np.linspace(o_data_y.min(), o_data_y.max(), 400)
        # create grid of eval points
        x_grid, y_grid = np.meshgrid(x_pts, y_pts)

        # eval kde on grid
        coords = np.vstack([x_grid.ravel(), y_grid.ravel()])
        kde_2d_vals = kde_2d(coords)
        kde_2d_vals = kde_2d_vals.reshape(x_grid.shape) # return to grid shape like done in kde2d_plotter

        # heatmap for simplicity
        hm = ax.imshow(kde_2d_vals,
                        origin='lower',
                        aspect='auto',
                        extent=[o_data_x.min(), o_data_x.max(), o_data_y.min(), o_data_y.max()],
                        cmap='Blues',
                        alpha=0.5
                        )
    ax.set_xlabel(name_x, fontsize=10)
    ax.set_ylabel(name_y, fontsize=10)
    ax.set_title(f'{name_x} vs {name_y} 2D KDE by {ori} orent and {factor_adj} factor', fontsize=12)
    plt.colorbar(hm, ax=ax, label='Density')
        
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

    # ----- QUESTION 2 -----
    # choosing a different bw factor for kde plot to achieve significantly different results
    # choose acc
    fig3, axs3 = plt.subplots(1, 2, figsize=(15,5))
    kde_plotter(acc, 'ACC', 'ACC', 'Density', axs3[0], 0.5)
    kde_plotter(acc, 'ACC', 'ACC', 'Density', axs3[1], 2.0)
    plt.tight_layout()
    plt.savefig('results/acc_kde_comp.png')

    # ----- QUESTION 3 -----
    # 2d histogram for amyg and acc
    fig5, ax5 = plt.subplots(figsize=(10,8))
    ax5.hist2d(amyg, acc, bins=10, density=True, cmap='Blues')
    ax5.set_title('Amygdala vs ACC 2D Histogram', fontsize=12)
    ax5.set_xlabel('Amygdala', fontsize=10)
    ax5.set_ylabel('ACC', fontsize=16)
    plt.colorbar(ax5.collections[0], ax=ax5, label='Density')
    plt.tight_layout()
    plt.savefig('results/amyg_acc_h2d.png')

    # ---- QUESTION 4 -----
    # 2d KDE for amyg and acc
    fig6, ax6 = plt.subplots(figsize=(10,8))
    kde2d_plotter(amyg, acc, 'Amygdala', 'ACC', ax6, 0.5)
    plt.tight_layout()
    plt.savefig('results/2d_kde.png')

    # ----- QUESTION 5 -----
    # conditional KDE for amyg by orientation
    fig7, ax7 = plt.subplots(figsize=(10,8))
    kde_cond_plotter(data, 'orientation', 'amygdala', ax7, 0.5)
    plt.tight_layout()
    plt.savefig('results/amyg_kde_by_orient.png')

    # conditional KDE for acc by orientation
    fig8, ax8 = plt.subplots(figsize=(10,8))
    kde_cond_plotter(data, 'orientation', 'acc', ax8, 0.5)
    plt.tight_layout()
    plt.savefig('results/acc_kde_by_orient.png')

    # write conditional means to csv
    cond_means = []
    sorted_orients = sorted(data['orientation'].unique())   
    for o in sorted_orients:
        cond_means.append({
            'orientation': o,
            'amygdala_mean': data[data['orientation'] == o]['amygdala'].mean(),
            'acc_mean': data[data['orientation'] == o]['acc'].mean()
        })
    cond_means_df = pd.DataFrame(cond_means)
    cond_means_df.to_csv('results/conditional_means.csv', index=False)

    # ----- QUESTION 6 -----
    # conditional 2d KDE for amyg and acc by orient
    fig9, ax9 = plt.subplots(2, 2, figsize=(16,12))
    # need to split into subplots to be able to differentiate on the heatmap
    ax9_flat = ax9.flatten() # 1D array req'd for imshow to work
    for idx, o in enumerate(sorted_orients):
        kde2d_cond_plotter(data, o, 'amygdala', 'acc', ax9_flat[idx], 0.5)
    plt.tight_layout()
    plt.savefig('results/2d_kde_by_orient.png')


    pass

if __name__ == "__main__":
    main()