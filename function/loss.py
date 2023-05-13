import numpy as np

# LOSS FUNCTION
def sse(target : float, output : float, derivative=False) -> float:
	""""Untuk linear, sigmoid, dan ReLU, gunakan fungsi loss berupa sum of squared error"""
	if(derivative):
		return (output - target)
	else:
		return ((target - output)**2) / 2

def cross_entropy(target : float, output : float, derivative=False) -> float:
	"""Untuk softmax, gunakan fungsi loss berupa cross entropy"""
	if(derivative):
		return -(target / output) + (1 - target) / (1 - output)
	else:
		return -(target * np.log(output))