import numpy as np

def sgd_update(X: np.ndarray, y: np.ndarray, weights: np.ndarray,
               learning_rate: float, n_iter: int) -> list:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.array(weights, dtype=float)
    n_samples = len(y)

    for i in range(n_iter):
        idx = i % n_samples              # cycle through samples in order
        xi, yi = X[idx], y[idx]
        pred = xi @ w                    # prediction for this one sample
        grad = 2 * (pred - yi) * xi      # per-sample MSE gradient
        w = w - learning_rate * grad     # gradient descent update

    return w.tolist()