import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

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
        # initial implementation wasn't correct, this should be fixed now
    # helper for cov init to ensure psd
    def init_cov(d):
        sig = np.random.randn(d, d)
        return sig @ sig.T + np.eye(d)
    sigma = [init_cov(d) for _ in range(c)]

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
            tau[:, k] = pi[k] * p_x * exp_term
        # norm the posteriors across comps and samples
        tau_sum = np.sum(tau, axis=1, keepdims=True) # sum across comps for each sample
        tau = tau / tau_sum # normalize

        # m-step updates
        # update pi
        N_k = np.sum(tau, axis=0) # sum across samples for each comp; (c,)
        pi = N_k / n # normalize

        # update mu
        for k in range(c):
            mu[k] = (tau[:,k:k+1].T @ data) / N_k[k].flatten() # weighted sum of samples for comp k
            # update sigma
            deltas = data - mu[k] # n x d
            wgtd_deltas = deltas * np.sqrt(tau[:, k:k+1])
            sigma[k] = (wgtd_deltas.T @ wgtd_deltas) / N_k[k]
        
        # log-likelihoods
        ll = np.zeros(n)
        for k in range(c):
            deltas = data - mu[k]
            inv = np.linalg.inv(sigma[k])
            det = np.linalg.det(sigma[k])
            p_x = 1.0 / np.sqrt((2 * np.pi) ** d * det)
            exp_term = np.exp(-0.5 * np.sum(deltas @ inv * deltas, axis=1))
            ll += pi[k] * p_x * exp_term
        ll = np.sum(np.log(ll + 1e-10)) # avoid log(0)

        log_liks.append(ll)

        print(f"iter {i + 1}, log-likelihood: {ll:.2f}")

        # check for convergence
        if len(log_liks) > 1 and abs(log_liks[-1] - log_liks[-2]) < tol:
            print("Converged within tol")
            break
    return mu, sigma, pi, tau, log_liks

def misclass_rates(true_labels, pred_labels):

    # 6 is true class
    true_binary = (true_labels == 6).astype(int)

    # accuracies for both labelings
    accuracy1 = np.mean(pred_labels == true_binary)
    accuracy2 = np.mean(pred_labels == (1-true_binary))
    # if one accuracy is worse than the other, pick the appropriate label that aligns
    if accuracy2 > accuracy1:
        pred_labels = 1 - pred_labels

    # counts per class
    class0counts = np.sum(true_binary == 0)
    class1counts = np.sum(true_binary == 1)

    # misclassification rates
    mc_rate_0 = np.sum((pred_labels == 1) & (true_binary == 0)) / class0counts
    mc_rate_1 = np.sum((pred_labels == 0) & (true_binary == 1)) / class1counts
    overall_mc = np.mean(pred_labels != true_binary)

    # confusion matrix
    conf_matrix = confusion_matrix(true_binary, pred_labels)

    return mc_rate_0, mc_rate_1, overall_mc, conf_matrix


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

    # em gmm implementation
    mu, sigma, pi, tau, log_liks = em_gmm(dataT_pca, comps=2, i_max=100, tol=1e-3)

    # ----- question 1 reporting -----
    # plot log-likelihoods
    plt.figure(figsize=(12, 9))
    plt.plot(range(1, len(log_liks) + 1), log_liks)
    plt.xlabel("Iter")
    plt.ylabel("Log-Likelihood")
    plt.title("Log-Likelihood vs Iter")
    plt.savefig("results/ll_iter.png")

    # ----- question 2 reporting -----
    # weights
    res_summary = pd.DataFrame({
        'Component': [1, 2],
        'Weight': pi,
        'Avg_PC1': mu[:, 0],
        'Avg_PC2': mu[:, 1],
        'Avg_PC3': mu[:, 2],
        'Avg_PC4': mu[:, 3]
    })
    res_summary.to_csv('results/gmm_summary.csv', index=False)

    # map back to orig space as 28x28
    mu_og = pca.inverse_transform(mu) # 2 x 784
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 6))
    for k in range(len(mu)):
        mean_img = mu_og[k].reshape(28, 28)
        axes2[k].imshow(mean_img, cmap='gray')
        axes2[k].set_title(f'Component {k+1} Mean\n(π = {pi[k]:.4f})')
        axes2[k].axis('off')

    plt.savefig("results/gmm_means.png")

    # heat maps
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 6))
    for k in range(len(sigma)):
        im = axes3[k].imshow(sigma[k], cmap='gray', aspect='auto')
        axes3[k].set_title(f'Component {k+1} Covariance')
        plt.colorbar(im, ax=axes3[k])
    plt.savefig("results/gmm_covs_hms.png")

    # ----- question 3 reporting -----
    gmm_tau = np.argmax(tau, axis=1)

    # kmeans
    kmeans = KMeans(n_clusters=2, random_state=19, n_init=12)
    kmeans_labels = kmeans.fit_predict(dataT_pca)

    # misclassification rates for gmm
    gmm_mc_2, gmm_mc_6, gmm_overall, gmm_cm = misclass_rates(labelsF, gmm_tau)
    # misclassification rates for kmeans
    kmeans_mc_2, kmeans_mc_6, kmeans_overall, kmeans_cm = misclass_rates(labelsF, kmeans_labels)

    # save confusion matrices
    pd.DataFrame(gmm_cm,
                index=['True 2', 'True 6'],
                columns=['Pred 2', 'Pred 6']    
                 ).to_csv("results/gmm_confusion_matrix.csv")
    pd.DataFrame(kmeans_cm,
                index=['True 2', 'True 6'],
                columns=['Pred 2', 'Pred 6']
                 ).to_csv("results/kmeans_confusion_matrix.csv")

    # misclassification rate gmm vs kmeans
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.bar(['K-Means', 'GMM'], [kmeans_overall, gmm_overall])
    ax1.set_title('Overall Misclassification Rate')
    ax1.set_ylabel('Misclassification Rate')
    ax1.set_ylim([0, max(kmeans_overall, gmm_overall) * 1.2])

    # accuracy rate gmm vs kmeans
    ax2.bar(['K-Means', 'GMM'], [1-kmeans_overall, 1-gmm_overall])
    ax2.set_title('Overall Accuracy Rate')
    ax2.set_ylabel('Accuracy Rate')
    ax2.set_ylim([0, 1])
    plt.savefig("results/mc_accuracy_comparison.png")

    # csv comparison
    comp_summary = pd.DataFrame({
        'Method': ['K-Means', 'GMM'],
        '2_Misclass': [kmeans_mc_2, gmm_mc_2],
        '6_Misclass': [kmeans_mc_6, gmm_mc_6],
        'Overall_Misclass': [kmeans_overall, gmm_overall],
        'Overall_Accuracy': [1-kmeans_overall, 1-gmm_overall]
    })

    comp_summary.to_csv("results/comparison_summary.csv", index=False)

if __name__ == "__main__":
    main()