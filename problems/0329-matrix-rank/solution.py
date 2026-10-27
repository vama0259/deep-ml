import numpy as np

def matrix_rank(A: np.ndarray, tol: float = 1e-10) -> int:
    M = np.array(A, dtype=float)   # copy, so we don't modify the input
    if M.ndim == 1:
        M = M.reshape(1, -1)
    m, n = M.shape

    rank = 0          # also the row where the next pivot goes
    for col in range(n):
        if rank == m:                 # no rows left
            break

        # find the largest-magnitude entry in this column, at or below row `rank`
        pivot_row = rank + np.argmax(np.abs(M[rank:, col]))

        if abs(M[pivot_row, col]) <= tol:
            continue                  # no pivot in this column, try the next one

        # swap the pivot row up
        M[[rank, pivot_row]] = M[[pivot_row, rank]]

        # zero out everything below the pivot
        for r in range(rank + 1, m):
            factor = M[r, col] / M[rank, col]
            M[r, col:] -= factor * M[rank, col:]

        rank += 1

    return rank