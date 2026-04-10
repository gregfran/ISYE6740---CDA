import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
os.makedirs('results', exist_ok=True)

path = os.path.join(script_dir, "data", "spambase.data")
df = pd.read_csv(path, header=None)  # add sep="," or sep=r"\s+" if needed

# response var is last col
y = df.iloc[:, -1].values.astype(int)
X = df.iloc[:, :-1].values

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6740)

# CART
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
tree_depth = cart.get_depth()
num_leaves = cart.get_n_leaves()
print(f"test acc {tree_acc:.3f}")
print(f"tree depth: {tree_depth}, leaves: {num_leaves}")

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
plt.show()