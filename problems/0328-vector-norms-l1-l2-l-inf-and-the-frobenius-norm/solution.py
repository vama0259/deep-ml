import numpy as np

def compute_norm(arr: np.ndarray, norm_type: str) -> float:
    """
    Compute the specified norm of the input array.

    'l1', 'l2' and 'linf' are entrywise norms and accept a 1D or 2D array.
    'frobenius' is a matrix norm and must raise a ValueError if arr is not 2D.

    Args:
        arr: Input numpy array (1D or 2D)
        norm_type: Type of norm ('l1', 'l2', 'linf', or 'frobenius')

    Returns:
        The computed norm as a float
    """

    arr = arr.astype(float)

    
    if norm_type == 'l1':
        arr = np.abs(arr)
        return np.sum(arr)
    elif norm_type == 'l2':
        value = np.sum(arr ** 2)
        return value ** 0.5
    elif norm_type == 'linf':
        value = np.max(np.abs(arr))
        return value
    elif norm_type == 'frobenius':
        if arr.ndim == 1:
             raise ValueError("Expected a 2D array")
        else:
            return np.sqrt(np.sum(arr ** 2))