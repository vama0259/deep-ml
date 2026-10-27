import numpy as np

def is_linearly_independent(vectors: list[list[float]]) -> bool:
    """
    Check if a set of vectors is linearly independent.
    
    Args:
        vectors: List of vectors, where each vector is a list of floats.
                 All vectors must have the same dimension.
        
    Returns:
        True if vectors are linearly independent, False otherwise.
    """
    # Your cod
    arr = np.array(vectors, dtype=float)

    if arr.size == 0:
        return True

    m, n = arr.shape
    if m > n:
        return False

    rank = np.linalg.matrix_rank(arr)
    return bool(rank == m)