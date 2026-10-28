
def orthogonal_projection(v, L):
	"""
	Compute the orthogonal projection of vector v onto line L.

	:param v: The vector to be projected
	:param L: The line vector defining the direction of projection
	:return: List representing the projection of v onto L
	"""
	import numpy as np
	V = np.array(v)
	L = np.array(L)

	mag = np.dot(V, L)/ np.dot(L,L)


	return (mag*L).tolist()
