def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	import numpy as np 
	A = np.array(a)
	B = np.array(b)
	if A.shape[1] != B.shape[0]:
		return -1
	ans = np.dot(a, b)
	return ans.tolist()