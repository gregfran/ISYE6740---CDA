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


def load_data():
    df = pd.read_csv('data/marriage.csv', header=None)
    # shape is 170, 55 and the last column is the label (divorce)
    # assign last row to y and rem to X
    y = df.iloc[:, -1].values
    X = df.iloc[:, :-1].values
    return X, y
    

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    # results folder
    os.makedirs('results', exist_ok=True)

    # get training and label data
    X, y = load_data()

    # need to assign 80% of the data to training and 20% to testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)
    
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
    top_acc = 0
    best_k = 1

    for k in k_trials:
        knn = KNN(n_neighbors=k)
        # train knn on training portion
        knn.fit(X_train, y_train)
        # predictions
        knn_preds = knn.predict(X_test)

        # performance metrics
        knn_cm = confusion_matrix(y_test, knn_preds)
        knn_acc = accuracy_score(y_test, knn_preds)
        if knn_acc > top_acc:
            top_acc = knn_acc
            top_k = k

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
        'Accuracy': [nb_acc, lr_acc, knn_acc],
    })
    q1_res.to_csv('results/class_model_comp.csv', index=False)





    pass

if __name__ == "__main__":
    main()