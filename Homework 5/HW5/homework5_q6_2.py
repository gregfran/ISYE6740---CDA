import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
os.makedirs('results', exist_ok=True)
np.random.seed(6740)

data = loadmat("data/cs.mat")
print(data.keys())

img = data["img"] # 50 x 50
# print("img shape:", img.shape)
x_flat = img.flatten() # 2500,
# print(x_flat.shape)

num_pixels = x_flat.shape[0]
n = 1300

# y = Ax + eps
# calc A
A = np.random.normal(0, 1, size=(n, num_pixels))
# calc eps
# std dev is input for random.normal so need to use 5 == sqrt(25)
epsilon = np.random.normal(0, 5, size=n)
# calc y
y = A @ x_flat + epsilon

# lambda range, keeping general for now
lambdas = np.logspace(-2, 4, 50)
k_fold = KFold(n_splits=10, shuffle=True, random_state=6740)

avg_mse = []
std_mse = []

for lam in lambdas:
    ridge = Ridge(alpha=lam, fit_intercept=False)
    scores = cross_val_score(
        ridge, A, y,
        cv=k_fold,
        scoring="neg_mean_squared_error"
    )
    mse = -1*scores
    avg_mse.append(np.mean(mse))
    std_mse.append(np.std(mse))

avg_mse = np.array(avg_mse)
std_mse = np.array(std_mse)

# selected lambda
best_idx = np.argmin(avg_mse)
lambda_best = lambdas[best_idx]

print("best lambda:", lambda_best)

ridge_best = Ridge(alpha=lambda_best, fit_intercept=False)
ridge_best.fit(A, y)

x_hat = ridge_best.coef_
img_hat = x_hat.reshape(50, 50)

plt.figure(figsize=(10, 5))
plt.semilogx(lambdas, avg_mse, marker='o')
plt.fill_between(
    lambdas,
    avg_mse - std_mse,
    avg_mse + std_mse,
    alpha=0.2
)
plt.axvline(lambda_best, linestyle=':', label=f"best lambda = {lambda_best:.4f}")
plt.xlabel("lambda")
plt.ylabel("Cross-Val MSE (k=10)")
plt.title("Ridge Error Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("results/ridge_cv_curve.png")

# recovery plot
plt.figure(figsize=(10, 4))

# actual plot
plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Actual Image")
plt.axis("off")

# recovered plot
plt.subplot(1, 2, 2)
plt.imshow(img_hat, cmap="gray")
plt.title(f"Recovered Img (lambda = {lambda_best:.4f})")
plt.axis("off")

plt.tight_layout()
plt.savefig("results/ridge_recovery.png")