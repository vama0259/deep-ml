import numpy as np

def mini_batch_gd_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray,
                       bias: float, batch_indices: list, lr: float) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(weights, dtype=float)

    Xb = X[batch_indices]                 # rows chosen for this mini-batch
    yb = y[batch_indices]
    m = len(batch_indices)

    preds = Xb @ w + bias                 # prediction for every row in the batch
    err = preds - yb                      # error for every row

    grad_w = (2 / m) * Xb.T @ err         # gradient for weights
    grad_b = (2 / m) * np.sum(err)        # gradient for bias

    w_new = w - lr * grad_w
    b_new = bias - lr * grad_b

    return np.append(w_new, b_new)        # [w..., b]