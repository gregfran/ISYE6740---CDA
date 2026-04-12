import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler


script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
os.makedirs('results', exist_ok=True)

path = os.path.join(script_dir, "data", "spambase.data")
df = pd.read_csv(path, header=None)  # add sep="," or sep=r"\s+" if needed

# response var is last col
y = df.iloc[:, -1].values.astype(int)
X = df.iloc[:, :-1].values

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=6740, shuffle=True, stratify=y)

# ----> CART
cart = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=20,
    random_state=6740
)

# fit model
cart.fit(X_train, y_train)

# performance
y_pred = cart.predict(X_test)
tree_acc = accuracy_score(y_test, y_pred)
tree_error = np.mean(y_pred != y_test)
tree_depth = cart.get_depth()
num_leaves = cart.get_n_leaves()
print(f"cart test acc {tree_acc:.3f}")
print(f"cart test error {tree_error:.3f}")
print(f"cart tree depth: {tree_depth}, leaves: {num_leaves}")

names = ["non-spam", "spam"]

plt.figure(figsize=(28, 16))
plot_tree(
    cart,
    class_names=names,
    rounded=True,
    fontsize=8,
    impurity=False,
    proportion=True,
    precision=2,
    max_depth=3 # too much depth and it gets way too crowded
)
plt.title("CART (spambase)", fontsize=16)
plt.tight_layout()

save_path = os.path.join(script_dir, "results", "cart_spambase.png")
plt.savefig(save_path)
# plt.show()

# -----> Random Forest
tree_grid = list(range(1, 201, 4))
ranf_errors = []

for tree in tree_grid:
    ranf = RandomForestClassifier(
        n_estimators=tree,
        max_depth=5,
        random_state=6740,
    )
    ranf.fit(X_train, y_train)
    y_pred_ranf = ranf.predict(X_test)
    ranf_errors.append(np.mean(y_pred_ranf != y_test))

rf_last_error = ranf_errors[-1]
print(f"Random Forest n trees: {tree_grid[-1]}")
print(f"Random Forest test error: {rf_last_error:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(tree_grid, ranf_errors, marker='o', markersize=3, linewidth=1.8, label='Random Forest test error')
plt.axhline(y=tree_error, color='red', linestyle='--', linewidth=2, label='CART test error')
plt.xlabel("Num Trees")
plt.ylabel("Test Error (Misclassification)")
plt.title("Test Error vs Num Trees: Random Forest vs CART")
plt.grid(alpha=0.5)
plt.legend()
plt.tight_layout()

ranf_path = os.path.join(script_dir, "results", "ranf_cart_test_error.png")
plt.savefig(ranf_path, bbox_inches="tight")
# plt.show()

# -----> sensitivity exploration

f = X_train.shape[1] # features
vals = list(range(1, f+1, 3))

oob_errs = []
test_errs = []

for val in vals:
    ranf_se = RandomForestClassifier(
        n_estimators=400,
        max_features=val,
        oob_score=True,
        bootstrap=True,
        random_state=6740,
    )
    ranf_se.fit(X_train, y_train)
    # errors
    # oob
    oob_errs.append(1 - ranf_se.oob_score_)

    # test misclass
    y_pred_se = ranf_se.predict(X_test)
    test_errs.append(np.mean(y_pred_se != y_test))

best_oob_ind = np.argmin(oob_errs)
best_oob_val = vals[best_oob_ind]
best_oob_err = oob_errs[best_oob_ind]

print(f"best oob val: {best_oob_val}, best oob error: {best_oob_err:.4f}")
print(f"corresponding test error: {test_errs[best_oob_ind]:.4f}")

plt.figure(figsize=(10, 6), dpi=160)
plt.plot(vals, oob_errs, marker='o', markersize=3, linewidth=1.8, label='OOB error')
plt.plot(vals, test_errs, marker='s', markersize=3, linewidth=1.8, label='Test error')
plt.xlabel(r"Num Features")
plt.ylabel("Misclassification Error")
plt.title(r"Random Forest Sensitivity to no. features")
plt.grid(alpha=0.5)
plt.legend()
plt.tight_layout()

sens_path = os.path.join(script_dir, "results", "rf_oob_test.png")
plt.savefig(sens_path)
# plt.show()

# -----> 1 Class SVM
# need to shuffle and split .75/.25
X_train_ocsvm, X_test_ocsvm, y_train_ocsvm, y_test_ocsvm = train_test_split(X, y, test_size=0.25, random_state=6740, shuffle=True, stratify=y)

# need to use only label 0 (non spam) for training
# extract all non-spam emails from the training block
X_train_ocsvm_nonspam = X_train_ocsvm[y_train_ocsvm == 0]

# validation set for tuning
X_train_ocsvm_spam = X_train_ocsvm[y_train_ocsvm == 1]

# split training block (non spam one) for fit and validation
# using 80/20 for this
X_fit_nonspam, X_val_nonspam = train_test_split(X_train_ocsvm_nonspam, test_size=0.25, random_state=6740, shuffle=True)

# need to create the validation set for tuning by combining the val non spam and the spam portion of the training block
X_val_ocsvm = np.vstack((X_val_nonspam, X_train_ocsvm_spam))
y_val_ocsvm = np.hstack((np.ones(X_val_nonspam.shape[0]), -np.ones(X_train_ocsvm_spam.shape[0])))
# scale
oscvm_scaler = StandardScaler()
X_fit_nonspam_scaled = oscvm_scaler.fit_transform(X_fit_nonspam)
X_val_ocsvm_scaled = oscvm_scaler.transform(X_val_ocsvm)

# TUNING
# init conds
best_error = np.inf
best_gamma = None
best_nu = None
# grid search parms
gammas = [0.001, 0.01, 0.1, 1]
nus = [0.01, 0.05, 0.1, 0.5]

for nu in nus:
    for gam in gammas:
        ocsvm = OneClassSVM(kernel='rbf', gamma=gam, nu=nu)
        ocsvm.fit(X_fit_nonspam_scaled)
        # predict on val set
        y_val_pred = ocsvm.predict(X_val_ocsvm_scaled)
        # misclass error
        val_error = np.mean(y_val_pred != y_val_ocsvm)
        if val_error < best_error:
            best_error = val_error
            best_gamma = gam
            best_nu = nu

print(f"best gamma: {best_gamma}, best nu: {best_nu}, best val error: {best_error:.4f}")

# need to use the best tuning params for the final fit
X_train_ocsvm_nonspam_scaled = oscvm_scaler.fit_transform(X_train_ocsvm_nonspam)
ocsvm_final = OneClassSVM(kernel='rbf', gamma=best_gamma, nu=best_nu)
ocsvm_final.fit(X_train_ocsvm_nonspam_scaled)
# predict on test set
X_test_ocsvm_scaled = oscvm_scaler.transform(X_test_ocsvm)
y_test_pred = ocsvm_final.predict(X_test_ocsvm_scaled)

# need to remap because of -1 and 1 labels from before
y_test_ocsvm_mapped = np.where(y_test_ocsvm == 0, 1, -1)
ocsvm_test_error = np.mean(y_test_pred != y_test_ocsvm_mapped)
print(f"one class svm test error: {ocsvm_test_error:.4f}")