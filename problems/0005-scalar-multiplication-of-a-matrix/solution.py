def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
	import numpy as np
	A = np.array(matrix)
	output = scalar * A
	return output.tolist()