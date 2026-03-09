import numpy as np
import pandas as pd
import os
# sklearn pkgs
# data processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# naive bayes
from sklearn.naive_bayes import GaussianNB as GNB
# log reg
from sklearn.linear_model import LogisticRegression as LREG
# knn
from sklearn.neighbors import KNeighborsClassifier as KNN
# performance stuff
from sklearn.metrics import confusion_matrix, accuracy_score
# PCA
from sklearn.decomposition import PCA
# matplotlib
import matplotlib.pyplot as plt


def load_data():
    df = pd.read_csv('data/marriage.csv', header=None)
    # no headers
    # shape is 170, 55 and the last column is the label (divorce)
    # assign last row to y and rem to X
    y = df.iloc[:, -1].values.astype(int)  # int for binary classifications
    X = df.iloc[:, :-1].values.astype(float) # ensure float

    return X, y

def decision_boundary(model, X, y, ax, title):
    # mesh grid for plot
    # get the limits of data for plotting
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1 # 1 for padding
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1 
    # mesh
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    # predict on mesh and plot
    P = model.predict(np.c_[xx.flatten(), yy.flatten()]).reshape(xx.shape)
    ax.contourf(xx, yy, P, alpha=0.3, cmap='cool') # boundary fill
    ax.contour(xx, yy, P, colors='green', linewidths=1.0)  # bound line
    # scatter data points
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='cool', edgecolor='black', s=30)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('PC1', fontsize=10)
    ax.set_ylabel('PC2', fontsize=10)
    
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    # results folder
    os.makedirs('results', exist_ok=True)

    # get training and label data
    X, y = load_data()

    # need to assign 80% of the data to training and 20% to testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=19, stratify=y)
    
    # scale data to avoid certain features outweighing others
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # models for performance comparisons
    # naive bayes, log reg, knn

    # --> naive bayes
    nb = GNB()
    # train nb on training portion
    nb.fit(X_train, y_train)
    # predictions
    nb_preds = nb.predict(X_test)

    # performance metrics
    nb_cm = confusion_matrix(y_test, nb_preds)
    nb_acc = accuracy_score(y_test, nb_preds)

    # --> log reg
    lr = LREG()
    # train lr on training portion
    lr.fit(X_train, y_train)
    # predictions
    lr_preds = lr.predict(X_test)

    # performance metrics
    lr_cm = confusion_matrix(y_test, lr_preds)
    lr_acc = accuracy_score(y_test, lr_preds)

    # --> knn
    # need to figure out the best k cluster amt
    k_trials = range(1, 25)
    knn_accs= []
    top_k = 1
    top_acc = 0

    for k in k_trials:
        knn = KNN(n_neighbors=k)
        # train knn on training portion
        knn.fit(X_train, y_train)
        # predictions
        knn_preds = knn.predict(X_test)

        # performance metrics
        knn_cm = confusion_matrix(y_test, knn_preds)
        knn_acc = accuracy_score(y_test, knn_preds)
        knn_accs.append(knn_acc)
        if knn_acc > top_acc:
            top_acc = knn_acc
            top_k = k
        print(k)
        print(knn_acc)

    # train knn with best k
    knn = KNN(n_neighbors=top_k)
    knn.fit(X_train, y_train)
    knn_preds = knn.predict(X_test)
    print(top_k)

    # performance metrics
    knn_cm = confusion_matrix(y_test, knn_preds)
    knn_acc = accuracy_score(y_test, knn_preds)

    # --> q1 results
    q1_res = pd.DataFrame({
        'Method': ['Naive Bayes', 'Logistic Regression', f'KNN (k={top_k})'],
        'Accuracy': [f"{nb_acc:.4f}", f"{lr_acc:.4f}", f"{knn_acc:.4f}"],
    })
    q1_res.to_csv('results/class_model_comp.csv', index=False)

    # ----- Q2 PCA -----
    # data prep, assignment calls for 2 comps
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # --> naive bayes pca
    nb_pca = GNB()
    nb_pca.fit(X_train_pca, y_train)
    nb_pca_preds = nb_pca.predict(X_test_pca)
    # pca accuracy for nb
    nb_pca_acc = accuracy_score(y_test, nb_pca_preds)

    # --> log reg pca
    lr_pca = LREG()
    lr_pca.fit(X_train_pca, y_train)
    lr_pca_preds = lr_pca.predict(X_test_pca)
    # pca accuracy for lr
    lr_pca_acc = accuracy_score(y_test, lr_pca_preds)

    # --> knn pca
    top_acc_pca = 0
    top_k_pca = 1
    knn_accs_pca = []


    for k in k_trials:
        knn_pca = KNN(n_neighbors=k)
        knn_pca.fit(X_train_pca, y_train)
        knn_pca_preds = knn_pca.predict(X_test_pca)
        knn_pca_acc = accuracy_score(y_test, knn_pca_preds)
        knn_accs_pca.append(knn_pca_acc)
        if knn_pca_acc > top_acc_pca:
            top_acc_pca = knn_pca_acc
            top_k_pca = k

    # train knn with best k
    knn_pca = KNN(n_neighbors=top_k_pca)
    knn_pca.fit(X_train_pca, y_train)
    knn_pca_preds = knn_pca.predict(X_test_pca)

    # --> q2 results
    q2_res = pd.DataFrame({
        'Method': ['Naive Bayes', 'Logistic Regression', f'KNN (k={top_k_pca})'],
        'Accuracy': [f"{nb_pca_acc:.4f}", f"{lr_pca_acc:.4f}", f"{knn_pca_acc:.4f}"],
    })
    q2_res.to_csv('results/class_model_comp_pca.csv', index=False)

    # combine both sets of data for plots
    X_pca_all = np.vstack([X_train_pca, X_test_pca])
    y_all = np.concatenate([y_train, y_test])

    # decision boundary
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    decision_boundary(nb_pca, X_pca_all, y_all, ax1, f'Naive Bayes PCA')
    plt.tight_layout()
    plt.savefig('results/db_nb_pca.png')
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    decision_boundary(lr_pca, X_pca_all, y_all, ax2, f'Logistic Regression PCA')
    plt.tight_layout()
    plt.savefig('results/db_lr_pca.png')
    plt.show()

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    decision_boundary(knn_pca, X_pca_all, y_all, ax3, f'KNN k={top_k_pca} PCA')
    plt.tight_layout()
    plt.savefig('results/db_knn_pca.png')
    plt.show()

    pass

if __name__ == "__main__":
    main()