import math
import numpy as np

def _angle_between(a, b):
	""" Calculates the angle between two vectors safely.
	:param a: First input vector
	:type a: Iterable or numpy array of length 3
	:param b: Second input vector
	:type b: Iterable or numpy array of length 3
	:return: Angle
	:rtype: Radians
	"""
	a = np.copy(np.array(a))
	b = np.copy(np.array(b))
	if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
		raise ValueError('Got zero vector.')
	a /= np.linalg.norm(a)
	b /= np.linalg.norm(b)
	angle = np.arccos(np.dot(a, b))

	if np.isnan(angle):
		if np.isnan(np.min(a)) or np.isnan(np.min(b)):
			raise ValueError('NaN in vectors.')
		if (a == b).all():
			return 0.0
		else:
			return np.pi
	return angle

def distance_pbc(a, b, h_matrix):
	""" Calculates the distance between two vectors using periodic boundary conditions and the minimum image convention.
	:param a: First vector
	:type a: Numpy array of shape (3)
	:param b: Second vector
	:type b: Numpy array of shape (3)
	:param h_matrix: H matrix with the cell vectors as columns. Unit of length: Angstrom.
	:type h_matrix: Numpy array of shape (3, 3)
	:return: Distance between vectors.
	:rtype: Float
	"""
	hinv = np.linalg.inv(h_matrix)
	a_t = np.dot(hinv, a)
	b_t = np.dot(hinv, b)
	t_12 = b_t-a_t
	t_12 -= np.round(t_12)
	return np.linalg.norm(np.dot(h_matrix, t_12))

def hmatrix_to_abc(h_matrix, degrees=False):
	""" H matrix representation as box lengths and box angles.
	:param h_matrix: H matrix with the cell vectors as columns. Unit of length: Angstrom.
	:type h_matrix: Numpy array of shape (3, 3)
	:param degrees: Switch to return angles in degrees instead of radians.
	:type degrees: Boolean
	:return: Box specification following a, b, c, alpha, beta, gamma. Distances in Angstrom.
	:rtype: Numpy array of length 6
	"""
	result = np.zeros(6)
	for i in range(3):
		result[i] = np.linalg.norm(h_matrix[:, i])
	result[3] = _angle_between(h_matrix[:, 1], h_matrix[:, 2])
	result[4] = _angle_between(h_matrix[:, 0], h_matrix[:, 2])
	result[5] = _angle_between(h_matrix[:, 0], h_matrix[:, 1])
	if degrees:
		result[3:] = map(math.degrees, result[3:])
	return result

def abc_to_hmatrix(a, b, c, alpha, beta, gamma, degrees=True):
	""" Box vectors from box vector lengths and box vector angles.
	:param a: First box vector length in Angstrom.
	:type a: Float
	:param b: Second box vector length in Angstrom.
	:type b: Float
	:param c: Third box vector length in Angstrom.
	:type c: Float
	:param alpha: Fist box angle in radians.
	:type alpha: Float
	:param beta: Second box angle in radians.
	:type beta: Float
	:param gamma: Third box angle in radians.
	:type gamma: Float
	:param degrees: Switch to accept input angles in degrees.
	:type degrees: Boolean
	:return: H matrix with the cell vectors as columns. Unit of length: Angstrom.
	:rtype: Numpy array of shape (3, 3)
	"""
	if degrees:
		alpha, beta, gamma = map(math.radians, (alpha, beta, gamma))
	result = np.zeros((3, 3))

	a = np.array((a, 0, 0))
	b = b * np.array((math.cos(gamma), math.sin(gamma), 0))
	bracket = (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / math.sin(gamma)
	c = c * np.array((math.cos(beta), bracket, math.sqrt(math.sin(beta) ** 2 - bracket ** 2)))

	result[:, 0] = a
	result[:, 1] = b
	result[:, 2] = c

	return result

def repeat_vector(h_matrix, repeat_a, repeat_b, repeat_c):
	""" Builds a cartesian vector from a vector in the vector space created by the box vectors.
	:param h_matrix: H matrix with the cell vectors as columns. Unit of length: Angstrom.
	:type h_matrix: Numpy array of shape (3, 3)
	:param repeat_a: First coordinate in box vector coordinate system.
	:type repeat_a: Float
	:param repeat_b: Second coordinate in box vector coordinate system.
	:type repeat_b: Float
	:param repeat_c: Second coordinate in box vector coordinate system.
	:type repeat_c: Float
	:return: Cartesian vector. Unit of length: Angstrom.
	:rtype: Numpy array of length 3
	"""
	return h_matrix[:, 0] * repeat_a + h_matrix[:, 1] * repeat_b + h_matrix[:, 2] * repeat_c

def box_vertices(h_matrix, repeat_a, repeat_b, repeat_c):
	vertices = np.zeros((8, 3))
	vertices[0, :] = repeat_vector(h_matrix, repeat_a, repeat_b, repeat_c)
	vertices[1, :] = repeat_vector(h_matrix, repeat_a, repeat_b, repeat_c + 1)
	vertices[2, :] = repeat_vector(h_matrix, repeat_a, repeat_b + 1, repeat_c)
	vertices[3, :] = repeat_vector(h_matrix, repeat_a, repeat_b + 1, repeat_c + 1)
	vertices[4, :] = repeat_vector(h_matrix, repeat_a + 1, repeat_b + 1, repeat_c + 1)
	vertices[5, :] = repeat_vector(h_matrix, repeat_a + 1, repeat_b + 1, repeat_c)
	vertices[6, :] = repeat_vector(h_matrix, repeat_a + 1, repeat_b, repeat_c + 1)
	vertices[7, :] = repeat_vector(h_matrix, repeat_a + 1, repeat_b, repeat_c)
	return vertices


def vector_repetitions(minrval, h_matrix, index):
	""" Calculated the number of repetitions of a cell vector needed to extend the periodic system at least to a given radius.
	:param minrval: Minimum radius of the repeated system. Unit of length: Angstrom.
	:type minrval: Float
	:param h_matrix: H matrix with the cell vectors as columns. Unit of length: Angstrom.
	:type h_matrix: Numpy array of shape (3, 3)
	:param index: Index of the cell vector to use. Zero-based.
	:type index: Integer
	:return: Number of necessary repetitions of the selected cell vector.
	:rtype: Integer
	"""
	if index == 0:
		j1, j2 = 1, 2
	if index == 1:
		j1, j2 = 0, 2
	if index == 2:
		j1, j2 = 1, 0
	i = h_matrix[:, index]
	j1 = h_matrix[:, j1]
	j2 = h_matrix[:, j2]
	value = minrval/np.linalg.norm(i)
	value = max(value, minrval/np.linalg.norm(i+j1), minrval/np.linalg.norm(i+j2))
	value = max(value, 2*minrval/np.linalg.norm(i-j1), 2*minrval/np.linalg.norm(i-j2))
	return int(np.ceil(value))


def cell_volume(h_matrix):
	ab = np.cross(h_matrix[:, 0], h_matrix[:, 1])
	return np.abs(np.dot(ab, h_matrix[:, 2]))


def cartesian_to_scaled_coordinates(coordinates, h_matrix):
	h = np.linalg.inv(h_matrix)
	for i in range(len(coordinates)):
		coordinates[i] = (h * coordinates[i]).sum(axis=1)
	return coordinates


def scaled_to_cartesian_coordinates(coordinates, h_matrix):
	for i in range(len(coordinates)):
		coordinates[i] = (h_matrix * coordinates[i]).sum(axis=1)
	return coordinates


def cell_longest_diameter(h_matrix):
	""" Calculates the length of the longest vector possible within a unit cell for a given H matrix.
	:param h_matrix: H matrix with the cell vectors as columns. Unit of length: Angstrom.
	:type h_matrix: Numpy array of shape (3, 3)
	:return: Float
	"""
	a = h_matrix[:, 0]
	b = h_matrix[:, 1]
	c = h_matrix[:, 2]

	mdist = max(np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c))
	mdist = max(mdist, np.linalg.norm(a+b))
	mdist = max(mdist, np.linalg.norm(a+c))
	mdist = max(mdist, np.linalg.norm(c+b))
	mdist = max(mdist, np.linalg.norm(a+b+c))
	return mdist


def cell_multiply(coord, x, y, z, h_matrix=None, scaling_in=False, scaling_out=False):
	for i in (x, y, z):
		if i < 1 or int(i) != i:
			raise ValueError('Invalid image count.')

	# copy input
	coord = np.copy(coord)

	# prepare data
	factor = x * y * z
	atoms = coord.shape[0]
	newcoord = np.zeros((atoms * factor, 3))
	offset = 1

	# copy data
	if scaling_out:
		if not scaling_in:
			if h_matrix is None:
				raise TypeError('H matrix has to be given for partially cartesian data.')
			coord = cartesian_to_scaled_coordinates(coord, h_matrix)
		newcoord[:atoms] = coord

		for i_x in range(x):
			for i_y in range(y):
				for i_z in range(z):
					if i_x == i_y == i_z == 0:
						continue
					newcoord[atoms * offset:atoms * (offset + 1)] = coord
					newcoord[atoms * offset:atoms * (offset + 1), 0] += (i_x)
					newcoord[atoms * offset:atoms * (offset + 1), 1] += (i_y)
					newcoord[atoms * offset:atoms * (offset + 1), 2] += (i_z)
					offset += 1
		newcoord[:, 0] /= x
		newcoord[:, 1] /= y
		newcoord[:, 2] /= z
	else:
		if h_matrix is None:
			raise TypeError('H matrix has to be given for cartesian data.')
		if scaling_in:
			coord = scaled_to_cartesian_coordinates(coord, h_matrix)
		newcoord[:atoms] = coord
		for i_x in range(x):
			for i_y in range(y):
				for i_z in range(z):
					if i_x == i_y == i_z == 0:
						continue
					vector = h_matrix[:, 0] * i_x + h_matrix[:, 1] * i_y + h_matrix[:, 2] * i_z
					newcoord[atoms * offset:atoms * (offset + 1)] = coord + vector
					offset += 1

	return newcoord
