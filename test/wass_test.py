import ot
import numpy as np

def wass_cost(C):
	a = np.ones(C.shape[0])
	b = np.ones(C.shape[1])

	a /= a.sum()
	b /= b.sum()

	return ot.emd(a, b, C), ot.emd2(a, b, C)

C = np.array([
	[4.0, 4.0, 4.0]
])

print(wass_cost(C))

C = np.array([
	[3.7, 2.3, 1.7],
	[5.0, 3.7, 0.3]
])

print(wass_cost(C))

C = np.array([
	[0.0, 2.0, 8.0],
	[6.0, 4.0, 2.0],
	[8.0, 6.0, 0.0]
])

print(wass_cost(C))