import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient
		- direction: Unit vector in direction of steepest ascent
		- descent_direction: Unit vector in direction of steepest descent
	"""

	arr = np.array(gradient)
	mag = np.sqrt(np.sum(np.abs(arr)**2))
	if mag == 0:
		zeros = np.zeros(len(arr))
		return {'magnitude':mag, 'direction':zeros.tolist(), 'descent_direction':zeros.tolist()}
	direction = np.array([a/mag for a in gradient])
	desc_direction = -1 * direction
	return {'magnitude':mag, 'direction':direction.tolist(), 'descent_direction':desc_direction.tolist()}