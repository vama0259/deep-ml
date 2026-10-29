import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.array(weights, dtype=float)      # copy so the input isn't modified
    m = len(y)

    def grad(Xb, yb, w):
        # MSE gradient: (2 / rows) * X^T (X w - y)
        return (2 / len(yb)) * Xb.T @ (Xb @ w - yb)

    # how many samples go into each update
    if method == 'batch':
        size = m
    elif method == 'stochastic':
        size = 1
    elif method == 'mini_batch':
        size = batch_size
    else:
        raise ValueError(method)

    for _ in range(n_epochs):
        for start in range(0, m, size):      # consecutive, non-overlapping, no shuffling
            Xb = X[start:start + size]
            yb = y[start:start + size]
            w = w - learning_rate * grad(Xb, yb, w)

    return w