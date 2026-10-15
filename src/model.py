import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def cost(X, y, w, b, lam=0.0):
    m = len(y)
    f = np.clip(sigmoid(X @ w + b), 1e-12, 1 - 1e-12)
    loss = -np.mean(y * np.log(f) + (1 - y) * np.log(1 - f))
    return loss + lam / (2 * m) * np.sum(w ** 2)

def fit(X, y, lam=1.0, alpha=0.5, iters=3000):
    m, n = X.shape
    w, b, history = np.zeros(n), 0.0, []
    for i in range(iters):
        err = sigmoid(X @ w + b) - y
        w -= alpha * ((X.T @ err) / m + (lam / m) * w)
        b -= alpha * err.mean()
        if i % 50 == 0:
            history.append(cost(X, y, w, b, lam))
    return w, b, history

def predict_proba(X, w, b):
    return sigmoid(X @ w + b)