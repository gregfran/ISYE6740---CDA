import numpy as np
import pandas as pd
import os
from scipy.io import loadmat
import time
# knn
from sklearn.neighbors import KNeighborsClassifier as KNN
# performance stuff
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
# log reg
from sklearn.linear_model import LogisticRegression as LogReg
# nn
from sklearn.neural_network import MLPClassifier as NN
# svc
from sklearn.svm import SVC

def load_data(d_name=''):
    if d_name == "digits":
        # digits
        digits_data = loadmat('data/mnist_10digits.mat')
        digits_xtrain = digits_data['xtrain']
        digits_ytrain = digits_data['ytrain'].flatten()
        digits_xtest = digits_data['xtest']
        digits_ytest = digits_data['ytest'].flatten()

        return digits_xtrain, digits_ytrain, digits_xtest, digits_ytest

    elif d_name == "fashion":
        # fashion
        fashion_train = pd.read_csv('data/fashion-mnist_train.csv')
        fashion_test = pd.read_csv('data/fashion-mnist_test.csv')
        fashion_xtrain = fashion_train.iloc[:, 1:].values
        fashion_ytrain = fashion_train.iloc[:, 0].values
        fashion_xtest = fashion_test.iloc[:, 1:].values
        fashion_ytest = fashion_test.iloc[:, 0].values

        return fashion_xtrain, fashion_ytrain, fashion_xtest, fashion_ytest

    else:
        print("No dataset passed.")

def standardizer(xtrain, xtest):
    xtrain = xtrain.astype(np.float32) / 255.0
    xtest = xtest.astype(np.float32) / 255.0
    return xtrain, xtest

def evaluate_model(model, xtrain, ytrain, xtest, ytest, m_name, d_name):
    # simple time performance metrics for efficiency analysis discussion later
    start_train = time.time()
    model.fit(xtrain, ytrain)
    train_time = time.time() - start_train

    start_test = time.time()
    ypred = model.predict(xtest)
    test_time = time.time() - start_test

    acc = accuracy_score(ytest, ypred)
    # this func is easier than classification_report to extract results for csv
    prec, recall, f1, _ = precision_recall_fscore_support(ytest, ypred, average="weighted")

    return {
        "dataset": d_name,
        "model": m_name,
        "accuracy": acc,
        "precision": prec,
        "recall": recall,
        "f1": f1,
        "train_time_sec": train_time,
        "test_time_sec": test_time
    }

def knn_tuner(xtrain, ytrain, xtest, ytest, k_values):
    # need to figure out the best k cluster amt
    knn_accs= []
    top_k = 1
    top_acc = 0

    for k in k_values:
        knn = KNN(n_neighbors=k)
        # train knn on training portion
        knn.fit(xtrain, ytrain)
        # predictions
        knn_preds = knn.predict(xtest)

        # performance metrics
        knn_cm = confusion_matrix(ytest, knn_preds)
        knn_acc = accuracy_score(ytest, knn_preds)
        knn_accs.append(knn_acc)
        if knn_acc > top_acc:
            top_acc = knn_acc
            top_k = k

    # train knn with best k
    knn = KNN(n_neighbors=top_k)
    knn.fit(xtrain, ytrain)
    knn_preds = knn.predict(xtest)
    
    return top_k, top_acc, knn

def model_runner(xtrain, xtest, ytrain, ytest, d_name):

    xtrain, xtest = standardizer(xtrain, xtest)

    model_results = []

    # logreg
    log_reg = LogReg(
        random_state=6740,
        max_iter=1000,
        solver="lbfgs",
        multi_class="auto"
    )
    model_results.append(evaluate_model(log_reg, xtrain, ytrain, xtest, ytest, "LogReg", d_name))

    # knn
    k_values = range(1, 20, 2) # odd vals from 1 to 19 
    best_k, best_acc, _ = knn_tuner(xtrain, ytrain, xtest, ytest, k_values)
    knn = KNN(n_neighbors=best_k)
    model_results.append(evaluate_model(knn, xtrain, ytrain, xtest, ytest, f'KNN (k={best_k})', d_name))

    # NN
    neural_net = NN(
        hidden_layer_sizes=(20, 10),
        random_state=6740,
        max_iter=300
    )
    model_results.append(evaluate_model(neural_net, xtrain, ytrain, xtest, ytest, "NN (20,10)", d_name))

    # svm
    ran_state = np.random.default_rng(6740)
    ran_idx = ran_state.choice(len(xtrain), size=5000, replace=False)
    ran_xtrain = xtrain[ran_idx]
    ran_ytrain = ytrain[ran_idx]

        # linear
    svm_linear = SVC(kernel="linear", random_state=6740)
    model_results.append(evaluate_model(svm_linear, ran_xtrain, ran_ytrain, xtest, ytest, "SVM (linear)", d_name))

        # rbf kernel
    svm_rbf = SVC(kernel="rbf", random_state=6740)
    model_results.append(evaluate_model(svm_rbf, ran_xtrain, ran_ytrain, xtest, ytest, "SVM (rbf)", d_name))

    return model_results

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    # results folder
    os.makedirs('results', exist_ok=True)

    # set random seed
    np.random.seed(6740)

    for d_name in ["digits", "fashion"]:
        xtrain, ytrain, xtest, ytest = load_data(d_name)
        results = model_runner(xtrain, xtest, ytrain, ytest, d_name)
        res_df = pd.DataFrame(results)
        res_df.to_csv(f'results/model_res_{d_name}.csv', index=False)


    pass

if __name__ == "__main__":
    main()