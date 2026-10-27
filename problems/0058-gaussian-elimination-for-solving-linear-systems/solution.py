import numpy as np

def gaussian_elimination(A, b):
    """
    Solves Ax = b using Gaussian Elimination with partial pivoting.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)

    # forward elimination with partial pivoting
    for i in range(n):
        p = i + np.argmax(np.abs(A[i:, i]))   # row with the biggest pivot
        if p != i:
            A[[i, p]] = A[[p, i]]
            b[[i, p]] = b[[p, i]]
        for r in range(i + 1, n):
            factor = A[r, i] / A[i, i]
            A[r, i:] -= factor * A[i, i:]
            b[r] -= factor * b[i]

    # back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
    return x