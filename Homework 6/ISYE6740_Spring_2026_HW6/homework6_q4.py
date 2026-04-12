import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import KFold

# data paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
os.makedirs('results', exist_ok=True)

path = os.path.join(script_dir, "data", "data.mat")

# get data
mat = loadmat(path)
data = np.asarray(mat['data'], dtype=np.float64)

print("data shape:", data.shape)

# data shape: (201, 2), y is second col

y = data[:, 1]
X = data[:, [0]]

def weighted_reg(X_train, y_train, X_for_pred, h):
    len_train = len(X_train)
    # init prediction dataset to the size of X_for_pred
    preds = np.zeros(len(X_for_pred))
    x = X_train[:, 0] # get the x from the train set

    for idx in range(len(X_for_pred)):
        x0 = X_for_pred[idx, 0]
        x_dist = x-x0 # distance from the point of predict to pts in train set 

        w = np.exp(-0.5 * (x_dist/h)**2) # gaussian kernel weight calc

        w_sum = np.sum(w)
        
        preds[idx] = np.sum(w*y_train) / w_sum # weighted average of response (y) in train
    
    return preds

# h values
h_vals = np.logspace(-2, 1, 40)

kfold = KFold(n_splits=5, shuffle=True, random_state=6740)

cv_error_res = []

for h in h_vals:
    fold_errors = []

    for train_idx, test_idx in kfold.split(X):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        preds = weighted_reg(X_train, y_train, X_test, h)

        fold_error = np.mean((preds - y_test)**2) # mse (regression error)
        fold_errors.append(fold_error)
    cv_error_res.append(np.mean(fold_errors))

cv_error_res = np.array(cv_error_res)
best_h_idx = np.argmin(cv_error_res)
best_h = h_vals[best_h_idx]

print("best h:", best_h)
print("best cv error:", cv_error_res[best_h_idx])

# q4.1 plot cv error curve
plt.figure(figsize=(8, 5), dpi=160)
plt.semilogx(h_vals, cv_error_res, marker="o", linewidth=1.5)
plt.axvline(best_h, color="red", linestyle="--", label=f"best h = {best_h:.4g}")
plt.xlabel("h (log)")
plt.ylabel("cv-MSE")
plt.title("Weighted Linear Regression w. 5-fold CV")
plt.grid(alpha=0.5)
plt.legend()
wlr_path = os.path.join(script_dir, "results", "wlr_cv.png")
plt.savefig(wlr_path)
plt.show()

# q4.2 pred at x = 1.3 using best h
x_best_h = np.array([[1.3]])
y_pred_q2 = weighted_reg(X, y, x_best_h, best_h)[0]
print(f"pred y at x=1.3: {y_pred_q2}")

# pred curve over range
x_min = X[:, 0].min()
x_max = X[:, 0].max()
X_range = np.linspace(x_min, x_max, 200).reshape(-1, 1)
y_curve = weighted_reg(X, y, X_range, best_h)

# plt fit and pred at x=1.3
plt.figure(figsize=(8, 5), dpi=160)
plt.scatter(X[:, 0], y, s=18, alpha=0.7, label="Training data")
plt.plot(X_range[:, 0], y_curve, color="tab:orange", linewidth=2, label=f"Prediction curve (h={best_h:.4g})")
plt.scatter([1.3], [y_pred_q2], color="red", s=70, zorder=5, label=f"Prediction at x=1.3 (y={y_pred_q2:.4f})")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Weighted Linear Regression at x=1.3")
plt.grid(alpha=0.5)
plt.legend()

wlr_q42_path = os.path.join(script_dir, "results", "wlr_q42.png")
plt.savefig(wlr_q42_path)
plt.show()


