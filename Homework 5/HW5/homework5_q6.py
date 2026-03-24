import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat
from sklearn.linear_model import LassoCV, Lasso

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
lambdas = np.logspace(-2, 2, 50)

# lasso
lasso_cv = LassoCV(
    alphas=lambdas, # ed discussion said to treat lambda as alpha param for sklearn
    cv=10,
    fit_intercept=False,
    max_iter=10000,
    random_state=6740
)

lasso_cv.fit(A, y)

best_lam = lasso_cv.alpha_
x_hat = lasso_cv.coef_
img_hat = x_hat.reshape(50, 50)

print("best lambda:", best_lam)

# plot cv error curve
avg_mse = np.mean(lasso_cv.mse_path_, axis=1)
std_mse = np.std(lasso_cv.mse_path_, axis=1)

plt.figure(figsize=(10, 5))
plt.semilogx(lasso_cv.alphas_, avg_mse, marker='o')
plt.fill_between(
    lasso_cv.alphas_,
    avg_mse - std_mse,
    avg_mse + std_mse,
    alpha=0.2
)
plt.axvline(best_lam, linestyle=':', label=f"best lambda = {best_lam:.4f}")
plt.xlabel("lambda")
plt.ylabel("Cross-Val MSE (k=10)")
plt.title("Lasso Error Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("results/lasso_cv_curve.png")

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
plt.title(f"Recovered Img (lambda = {best_lam:.4f})")
plt.axis("off")

plt.tight_layout()
plt.savefig("results/lasso_recovery.png")
