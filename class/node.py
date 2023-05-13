import os
import sys

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..', 'function')
sys.path.append( mymodule_dir )

from loss import sse, cross_entropy

import numpy as np

class Node:
	def __init__(self, bias, weight, deltaWeight, activation_function_name, activation_function):
		self.bias = bias
		self.weight = weight
		self.sigma = 0
		self.activation_function = activation_function
		self.activation_function_name = activation_function_name
		self.activation_function_value = 0
		self.error = 0
		self.deltaBias = 0
		self.deltaWeight = deltaWeight
		self.deltaError = 0

	def calculate_sigma(self, inputData):
		self.sigma = np.dot(self.weight, inputData)
		self.sigma += self.bias
        
	def calculate_error(self, target):
		if (self.activation_function_name == "softmax"):
			self.error = cross_entropy(target, self.activation_function_value, derivative=False)
		else :
			self.error = sse(target, self.activation_function_value, derivative=False)

	def activate_neuron(self, sum = None):
		if (self.activation_function_name != 'softmax'):
			self.activation_function_value = self.activation_function(self.sigma)
		else:
			self.activation_function_value = self.activation_function(self.sigma, sum)

	def update_delta_bias(self, learning_rate):
		self.deltaBias += learning_rate * self.deltaError * 1
        
	def update_delta_weight(self, isFirstHiddenLayer, input_data, activation_function_value, learning_rate):
		for i in range (len(self.weight)):
			if(isFirstHiddenLayer):
				self.deltaWeight[i] += learning_rate * self.deltaError * input_data[i]
			else:
				self.deltaWeight[i] += learning_rate * self.deltaError * activation_function_value[i]
                
	def update_bias(self):
		self.bias = self.bias - self.deltaBias
        
	def update_weight(self):
		for i in range (len(self.weight)):
			self.weight[i] = self.weight[i] - self.deltaWeight[i]