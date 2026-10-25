def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
	import numpy as np
    A = np.array(a)
    B = np.array(b)
    if A.shape[1] != B.shape[0]:
        return -1
    else:
        c = A @ B
        return c.tolist()