def dice_statistics(n: int) -> tuple[float, float]:
	"""
	Compute the expected value and variance of a fair n-sided die roll.

	Args:
		n (int): Number of sides of the die

	Returns:
		tuple: (expected_value, variance)
	"""
	# Your code here
	import numpy as np
	arr = np.array(range(1,n+1))
	expecation = np.sum(arr) / n
	variance = np.sum((arr- expecation)**2) / n
	return (expecation, variance)