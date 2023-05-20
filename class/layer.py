import os
import sys
import random

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..', 'function')
sys.path.append( mymodule_dir )

from activation import linear, sigmoid, relu, softmax
from node import Node

class Layer:
	def __init__(self, n_neuron, activation_function_name, weights, bias, n_neuron_before_layer = None):
		activation_function = {
			'linear': linear,
			'sigmoid': sigmoid,
			'relu': relu,
			'softmax': softmax,
			'None' : None
		}
		self.n_neuron = n_neuron
		self.activation_function_name = activation_function_name
		self.activation_function = activation_function[activation_function_name]
		self.activation_function_value = []
		if ( n_neuron_before_layer != None ):
			self.n_neuron_before_layer = n_neuron_before_layer
			if (self.n_neuron_before_layer == 1): 
				self.weights = [random.uniform(0.0, 0.1) for i in range(self.n_neuron)]
			else: 
				self.weights = [[random.uniform(0.0, 0.1) for i in range(self.n_neuron)] for j in range(self.n_neuron_before_layer)]
			self.bias = [random.uniform(0.0, 0.1) for i in range(self.n_neuron)]
		else :
			self.weights = weights
			self.bias = bias
		self.net = []
		self.nodes = []
		self.generate_nodes()

	def generate_nodes(self):
		self.nodes = []
		for i in range(self.n_neuron):
			thisWeight = []
			thisDeltaWeight = []
			for j in range(len(self.weights)):
				thisWeight.append(self.weights[j][i])
				thisDeltaWeight.append(0)
			node = Node(self.bias[i], thisWeight, thisDeltaWeight, self.activation_function_name, self.activation_function)
			self.nodes.append(node)
            
	def update_layer(self):
		updateBias = []
		updateWeights = []
		for node in (self.nodes):
			updateBias.append(node.bias)
			j = 0
			for weights in (node.weight) :
				if (len(updateWeights) < len(node.weight)):
					updateWeights.append([weights])
				else:
					(updateWeights[j]).append(weights)
				j = j + 1
		self.bias = updateBias
		self.weights = updateWeights