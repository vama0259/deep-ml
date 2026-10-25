def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	import numpy as np
	if mode == 'row':
		mean = []
		for i in matrix:
			mean.append(sum(i)/float(len(i)))
		return mean
	elif mode == 'column':
		arr = np.array(matrix)
		arr = arr.T
		arr = arr.tolist()
		mean = []
		for i in arr:
			mean.append(sum(i)/float(len(i)))
		return mean