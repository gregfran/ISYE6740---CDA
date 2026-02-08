import numpy as np
import matplotlib.pyplot as plt
import os
from order_of_faces import OrderOfFaces

def nn_graph(adj_matrix, data):
    """
    adj matrix w/ samples
    """
    bi_adj = (adj_matrix > 0).astype(int)
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # adj_matrix plot
    im = ax.imshow(bi_adj, cmap='binary', interpolation='nearest')
    ax.set_title('Nearest Neighbor Adj Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('img idxx')
    ax.set_ylabel('img idxy')

    # sample faces
    n = data.shape[0]
    n_samples = 60
    face_size = 60
    ax.set_xlim(-1, n-1)
    ax.set_ylim(n-1, -1)
    ax.grid(True, alpha=0.3)

    k = int(np.ceil(np.sqrt(n_samples))) # to limit amt on each row for even distrbn
    rows = np.linspace(0, n-1, k, dtype=int)
    cols = np.linspace(0, n-1, k, dtype=int)
    cols = np.roll(cols, 5)

    # add faces at the nodes
    s = 0
    for i in rows:
        for j in cols:
            if s >= n_samples:
                break

            img = data[s].reshape(64, 64)

            x0, y0 = j, i
            # clip to keep on the plot
            x0 = np.clip(x0, -0.5 + face_size/2, n-0.5 - face_size/2)
            y0 = np.clip(y0, -0.5 + face_size/2, n-0.5 - face_size/2)
            x_l = x0 - face_size / 2
            x_r = x0 + face_size / 2
            y_t = y0 - face_size / 2
            y_b = y0 + face_size / 2
            
            ax.imshow(img,
                    extent=(x_l, x_r, y_b, y_t),
                    cmap='gray',
                    zorder=10)
            s += 1
        if s >= n_samples:
            break
    
    plt.savefig("results/nn_graph.png", bbox_inches='tight')
    plt.close()

def isomap_scat(embedding, data):
    fig, ax = plt.subplots(figsize=(16, 12))
    first_embd = embedding[:, 0]
    second_embd = embedding[:, 1]
    scat = ax.scatter(first_embd, second_embd, c='gray', alpha=0.6)

    # sample faces
    n = data.shape[0]
    n_samples = 60
    face_size = 6

    # embedding bounds for clipping
    x_min, x_max = first_embd.min(), first_embd.max()
    y_min, y_max = second_embd.min(), second_embd.max()
    ax.set_xlim(x_min - 1, x_max + 1)
    ax.set_ylim(y_min - 1, y_max + 1)

    k = int(np.ceil(np.sqrt(n_samples))) # like I did in nn_graph, to distribute the imgs

    # sample across the embeds
    rows = np.linspace(x_min, x_max, k)
    cols = np.linspace(y_min, y_max, k)
    cols = np.roll(cols, 3) 

    # add faces at grid positions in embedding space
    s = 0
    for y_pos in cols:
        for x_pos in rows:
            if s >= n_samples:
                break

            # find closest point in embed space to sample pos
            distances = (first_embd - x_pos)**2 + (second_embd - y_pos)**2
            idx = np.argmin(distances)
            
            img = data[idx].reshape(64, 64)

            x0, y0 = first_embd[idx], second_embd[idx]
            
            # clipping to keep on the plot
            x0 = np.clip(x0, x_min + face_size/2, x_max - face_size/2)
            y0 = np.clip(y0, y_min + face_size/2, y_max - face_size/2)
            
            x_l = x0 - face_size / 2
            x_r = x0 + face_size / 2
            y_t = y0 - face_size / 2
            y_b = y0 + face_size / 2
            
            ax.imshow(img,
                    extent=(x_l, x_r, y_b, y_t),
                    cmap='gray',
                    zorder=10)
            
            # coord labels for samples
            ax.text(x0, y_b + 1, f'({x0:.1f}, {y0:.1f})',
            ha='center', va='top', fontsize=6,
            color='red', fontweight='bold',
            zorder=11)
            s += 1
        if s >= n_samples:
            break
    
    ax.set_xlabel('ISOMAP idxx')
    ax.set_ylabel('ISOMAP idxy')
    ax.set_title('ISOMAP 2d Embedding')
    ax.grid(True, alpha=0.3)
    
    plt.savefig("results/isomap_embedding.png", bbox_inches='tight')
    plt.close()

def pca_scat(pca_result, data):
    fig, ax = plt.subplots(figsize=(16, 12))
    first_pc = pca_result[:, 0]
    second_pc = pca_result[:, 1]
    scat = ax.scatter(first_pc, second_pc, c='gray', alpha=0.6)

    # sample faces
    n = data.shape[0]
    n_samples = 60
    face_size = 2

    # embedding bounds for clipping
    x_min, x_max = first_pc.min(), first_pc.max()
    y_min, y_max = second_pc.min(), second_pc.max()

    ax.set_xlim(x_min - 1, x_max + 1)
    ax.set_ylim(y_min - 1, y_max + 1)

    k = int(np.ceil(np.sqrt(n_samples))) # like I did in nn_graph, to distribute the imgs

    # sample across the embeds
    rows = np.linspace(x_min, x_max, k)
    cols = np.linspace(y_min, y_max, k)
    cols = np.roll(cols, 4)

    # add faces at grid positions in embedding space
    s = 0
    for y_pos in cols:
        for x_pos in rows:
            if s >= n_samples:
                break

            # find closest point in embed space to sample pos
            distances = (first_pc - x_pos)**2 + (second_pc - y_pos)**2
            idx = np.argmin(distances)
            
            img = data[idx].reshape(64, 64)

            x0, y0 = first_pc[idx], second_pc[idx]
            
            # clipping to keep on the plot
            x0 = np.clip(x0, x_min + face_size/2, x_max - face_size/2)
            y0 = np.clip(y0, y_min + face_size/2, y_max - face_size/2)
            
            x_l = x0 - face_size / 2
            x_r = x0 + face_size / 2
            y_t = y0 - face_size / 2
            y_b = y0 + face_size / 2
            
            ax.imshow(img,
                    extent=(x_l, x_r, y_b, y_t),
                    cmap='gray',
                    zorder=10)
            
            # coord labels for samples
            ax.text(x0, y_b + 1, f'({x0:.1f}, {y0:.1f})',
            ha='center', va='top', fontsize=6,
            color='red', fontweight='bold',
            zorder=11)

            s += 1
        if s >= n_samples:
            break
    ax.set_xlabel('PCA idxx')
    ax.set_ylabel('PCA idxy')
    ax.set_title('PCA 2d Embedding')
    ax.grid(True, alpha=0.3)
    plt.savefig("results/pca_embedding.png", bbox_inches='tight')
    plt.close()


def main():
    dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(dir)
    os.makedirs('results', exist_ok=True)
    oof = OrderOfFaces()
    oof_data = oof.data
    best_eps = oof.best_eps
    adj_matrix = oof.get_adjacency_matrix(best_eps)
    nn_graph(adj_matrix, oof_data)
    embedding = oof.isomap(best_eps)
    isomap_scat(embedding, oof_data)
    num_dim = 2
    pca_result = oof.pca(num_dim)
    pca_scat(pca_result, oof_data)

if __name__ == "__main__":
    main()
