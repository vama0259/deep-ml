def vector_sum(a: list[int|float], b: list[int|float]) -> list[int|float]:
	# Return the element-wise sum of vectors 'a' and 'b'.
	# If vectors have different lengths, return -1.
	import numpy as np
	a = np.array(a)
	b = np.array(b)
	if a.size != b.size:
		return -1
	return a+b