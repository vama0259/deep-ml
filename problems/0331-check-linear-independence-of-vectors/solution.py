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
    # Your code here
    if len(vectors) == 0:
        return True
    arr = np.array(vectors)
    no_of_vectors , dimensions = arr.shape

    ''' if no_of_vectors > dimensions then it is dependented as if there are 2 columns x and y and we have 3 we can make 3rd using the 2 and 3 is not unique '''
    if no_of_vectors > dimensions:
        return False

    rank = np.linalg.matrix_rank(arr)

    return bool(rank==no_of_vectors) 


