import numpy as np
import matplotlib.pyplot as plt
import os
np.random.seed(6740)

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
# results folder
os.makedirs('results', exist_ok=True)

# data
n0 = 100
n1 = 50

x_bef = np.random.normal(loc=0, scale=np.sqrt(1), size=n0)
x_aft = np.random.normal(loc=0, scale=np.sqrt(1.4), size=n1)
x = np.concatenate([x_bef, x_aft])

# prep matrix for CUSUM calc
W = np.zeros(len(x))

# log likelihood ratio for N(0,1.4) vs N(0,1) as derived in the report
def log_likelihood_ratio(x):
    return -0.5 * np.log(1.4) + (x**2) / 7

for i in range(len(x)):
    W_prev = W[i - 1] if i > 0 else 0
    W[i] = max(0, W_prev + log_likelihood_ratio(x[i]))

# plot
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(W) + 1), W, label='CUSUM stat')
plt.axvline(x=108, linestyle='--', label='Change point (n0=108)', color='red')
plt.xlabel("idx")
plt.ylabel("CUSUM stat")
plt.title("CUSUM for N(0,1) to N(0,1.4)")
plt.legend()
plt.grid(True)
plt.savefig("results/cusum_plot.png")
plt.show()