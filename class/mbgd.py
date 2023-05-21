import os
import sys
import networkx as nx
import matplotlib.pyplot as plt

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..', 'function')
sys.path.append( mymodule_dir )

from loss import sse, cross_entropy

from layer import Layer

class MiniBatchGradientDescent:
	def __init__(self, max_iter : int, batch_size : int, error_threshold : float, learning_rate : float, n_layer = None, n_neuron_per_layer = None, activation_function_name_per_layer = None ):
		self.input_size        = 0
		self.input_layer       = []

		self.n_hidden_layer    = 0
		self.hidden_layer      = [] # berisi node hidden layer

		self.output_layer      = [] # berisi node output layer

		self.prediction_result = [] # berisi nilai output
		self.target            = [] # berisi target yang diharapkan
		self.error             = [] # berisi nilai error
		self.max_sse           = 0.00000001
		self.max_iter          = max_iter
		self.batch_size        = batch_size
		self.error_threshold   = error_threshold
		self.learning_rate     = learning_rate
		
		#Case not using file model
		if ( n_layer != None and n_neuron_per_layer != None and activation_function_name_per_layer != None ):
			self.n_layer = n_layer
			self.n_neuron_per_layer = n_neuron_per_layer
			for i in range (n_layer):
				if ( i != 0 ):
					if ( i == n_layer - 1 ):
						self.output_layer = Layer(self.n_neuron_per_layer[i], activation_function_name_per_layer[i], weights=None, bias=None, n_neuron_before_layer = self.n_neuron_per_layer[i-1], )
					else:
						self.hidden_layer.append(Layer(self.n_neuron_per_layer[i], activation_function_name_per_layer[i], weights=None, bias=None, n_neuron_before_layer = self.n_neuron_per_layer[i-1], ))
						self.n_hidden_layer += 1

	def set_input_data(self, X_train):
		self.input_layer = X_train
		self.input_size = len(X_train[0])

	def set_target(self, target):
		self.target = target

	def set_stopped_by(self, stopped_by):
		self.stopped_by = stopped_by
        
	def set_bias_final(self, bias_final):
		self.bias_final = bias_final
        
	def set_weights_final(self, weights_final):
		self.weights_final = weights_final

	def add_layer_node(self, layerType, input_size, n_neuron, activation, input_data, weights, bias):
		if (layerType == "input_layer") :
			self.input_layer = input_data
			self.input_size = input_size
		elif (layerType == "hidden_layer") :
			self.hidden_layer.append(Layer(n_neuron, activation, weights, bias))
			self.n_hidden_layer += 1
		elif (layerType == "output_layer") :
			self.output_layer = Layer(n_neuron, activation, weights, bias)

	def reset_net_and_activation_function_value(self, typeLayer):
		if (typeLayer == 'hidden layer'):
			for i in range (self.n_hidden_layer):
				self.hidden_layer[i].net = []
				self.hidden_layer[i].activation_function_value = []
		else:
			self.output_layer.net = []
			self.output_layer.activation_function_value = []

	def resetDelta(self):
		for i in range (self.n_hidden_layer):
			for node in (self.hidden_layer[i].nodes):
				node.deltaBias = 0
				for j in range (len(node.deltaWeight)):
					node.deltaWeight[j] = 0

		for node in (self.output_layer.nodes):
			node.deltaBias = 0
			for i in range (len(node.deltaWeight)):
				node.deltaWeight[i] = 0
                
	def resetError(self):
		self.error = []

	def sum_error(self):
		sumError = 0
		for i in range (len(self.error)):
			sumError = sumError + self.error[i]
		return(sumError)

	def get_error(self, activation_function_name, target, output_prediction, derivative=False):
		if (activation_function_name == "softmax"):
			return (cross_entropy(target, output_prediction, derivative))
		else :
			return (sse(target, output_prediction, derivative))
    
	def forward_pass(self, i):
		# hidden layers if any
		for j in range (self.n_hidden_layer) :
			# check is first hiddent layer or not
			if (j == 0):
				prevLayer_activationValues = self.input_layer[i]
			else:
				prevLayer_activationValues = self.hidden_layer[j-1].activation_function_value

			# reset net and activation function value if any
			if (len(self.hidden_layer[j].net) != 0 ):
				self.reset_net_and_activation_function_value('hidden layer')

			# calculate net
			for node in self.hidden_layer[j].nodes:
				node.calculate_net(prevLayer_activationValues)
				self.hidden_layer[j].net.append(node.net)
			
			# calculate activation function value
			for node in self.hidden_layer[j].nodes:
				# check if the activation function is softmax
				if (node.activation_function_name == 'softmax'):
					node.activate_neuron(self.hidden_layer[j].net)
				else:
					node.activate_neuron()
				self.hidden_layer[j].activation_function_value.append(node.activation_function_value)

		# output Layer
		# check if there is only a hidden layer or not
		if (self.n_hidden_layer == 0):
			prevLayer_activationValues = self.input_layer[i]
		else:
			prevLayer_activationValues = self.hidden_layer[-1].activation_function_value

		# reset net and activation function value if any
		if (len(self.output_layer.net) != 0 ):
				self.reset_net_and_activation_function_value('output layer')

		# calculate net
		for node in self.output_layer.nodes:
			node.calculate_net(prevLayer_activationValues)
			self.output_layer.net.append(node.net)

		# calculate activation function value
		for node in self.output_layer.nodes:
				# check if the activation function is softmax
			if (node.activation_function_name == 'softmax'):
				node.activate_neuron(self.output_layer.net)
			else:
				node.activate_neuron()
			self.output_layer.activation_function_value.append(node.activation_function_value)

		# calculate error
		k = 0
		for node in self.output_layer.nodes:
			node.calculate_error(self.target[i][k])
			self.error.append(node.error)
			k = k + 1
        
	def backward_pass(self, num):
		# output layer
		for i in range (len(self.output_layer.nodes)):
			# derivation of the error value to the output value
			dError_dOutput = self.get_error(self.output_layer.activation_function_name, self.target[num][i], self.output_layer.nodes[i].activation_function_value, derivative=True)

			# derivation of output value to input value (net)
			if (self.output_layer.nodes[i].activation_function_name == 'softmax'):
				dOutput_dInput = self.output_layer.nodes[i].activation_function(self.output_layer.nodes[i].net, self.output_layer.net, derivative=True)
			else:
				dOutput_dInput = self.output_layer.nodes[i].activation_function(self.output_layer.nodes[i].net, derivative=True)
        
			# update delta error values
			self.output_layer.nodes[i].deltaError = dError_dOutput * dOutput_dInput
        
		# hidden Layer
		for i in range (self.n_hidden_layer):
			# derivation of the error value to the output value
			dError_dOutput = []
			if (self.n_hidden_layer - 1 - i == self.n_hidden_layer - 1):
				for j in range (len(self.output_layer.nodes)):
					for k in range (len(self.output_layer.nodes[j].weight)):
						if (j == 0):
							dError_dOutput.append(self.output_layer.nodes[j].deltaError * self.output_layer.nodes[j].weight[k])
						else:
							dError_dOutput[k] = dError_dOutput[k] + self.output_layer.nodes[j].deltaError * self.output_layer.nodes[j].weight[k]
            
			else:
				for j in range (len(self.hidden_layer[-1 - i].nodes)):
					for k in range (len(self.hidden_layer[-1 - i].nodes[j].weight)):
						if (j == 0):
							dError_dOutput.append(self.hidden_layer[-1 - i].nodes[j].deltaError * self.hidden_layer[-1 - i].nodes[j].weight[k])
						else:
							dError_dOutput[k] = dError_dOutput[k] + self.hidden_layer[-1 - i].nodes[j].deltaError * self.hidden_layer[-1 - i].nodes[j].weight[k]

			for j in range (len(self.hidden_layer[i].nodes)):
				# derivation of output value to input value (net)
				if (self.output_layer.nodes[j].activation_function_name == 'softmax'):
					dOutput_dInput = self.hidden_layer[i].nodes[j].activation_function(self.hidden_layer[i].nodes[j].net, self.hidden_layer[i].net)
				else:
					dOutput_dInput = self.hidden_layer[i].nodes[j].activation_function(self.hidden_layer[i].nodes[j].net, derivative=True)

				# update delta error values
				self.hidden_layer[i].nodes[j].deltaError = dError_dOutput[j] * dOutput_dInput

                
	def updateDelta(self, i):
		# hidden layer
		for j in range (self.n_hidden_layer):
			for node in self.hidden_layer[j].nodes:
				node.update_delta_bias(self.learning_rate)
				if ( j == 0 ):
					node.update_delta_weight(True, self.input_layer[i], None, self.learning_rate)
				else:
					node.update_delta_weight(False, None, self.hidden_layer[j-1].activation_function_value, self.learning_rate)
                
		# output layer
		for node in self.output_layer.nodes:
			node.update_delta_bias(self.learning_rate)
			if ( self.n_hidden_layer == 0):
				node.update_delta_weight(True, self.input_layer[i], None, self.learning_rate)
			else:
				node.update_delta_weight(False, None, self.hidden_layer[-1].activation_function_value, self.learning_rate)

	def updateBiasWeight(self):
		# hidden layer
		for i in range (self.n_hidden_layer):
			for node in self.hidden_layer[i].nodes:
				node.update_bias()
				node.update_weight()
			self.hidden_layer[i].update_layer()
                
		# output layer
		for node in self.output_layer.nodes:
			node.update_bias()
			node.update_weight()
		self.output_layer.update_layer()

	def train(self):
		print("Akan dilakukan training")
		for i in range(self.max_iter):
			print(f"iter yang ke-{i+1}")
			idxReadInputData = 0
			while ( idxReadInputData != len(self.input_layer) ):
				for j in range(self.batch_size):
					print(f"batch size yang ke-{j+1}")
					self.forward_pass(idxReadInputData)
					self.backward_pass(idxReadInputData)
					self.updateDelta(idxReadInputData)
					idxReadInputData = idxReadInputData + 1
				print(f"lakukan update pada weight dan bias")
				self.updateBiasWeight()
				self.resetDelta()
			print(f"nilai error pada iter yang ke-{i+1} adalah {self.sum_error()}")
			if (self.sum_error() < self.error_threshold) :
				print(f"nilai {self.sum_error()} lebih kecil dari {self.error_threshold}")
				break
			elif (i == self.max_iter - 1):
				print("sudah mencapai max iter")
				break
			self.resetError()

	def information(self):
		if (self.n_hidden_layer != 0):
			print("")
			print(f"Berikut informasi yang ada pada hidden layer")
			for i in range(self.n_hidden_layer):
				print("")
				print(f"Pada hidden layer yang ke-{i+1} terdapat")
				for j in range (len(self.hidden_layer[i].nodes)):
					print(f"Pada node yang ke-{j+1} terdapat")
					print(f"Bias: {self.hidden_layer[i].nodes[j].bias}")
					print(f"Weight: {self.hidden_layer[i].nodes[j].weight}")

		print("")
		print(f"Berikut informasi yang ada pada output layer")
		for j in range (len(self.output_layer.nodes)):
			print(f"Pada node yang ke-{j+1} terdapat")
			print(f"Bias: {self.output_layer.nodes[j].bias}")
			print(f"Weight: {self.output_layer.nodes[j].weight}")

	def visualize(self):
		self.information()
		print()
		#set variable
		G = nx.Graph()
		pos = {}
		labels = {}
		counter = 1
		posCounter = 1
		inputNode = []
		hiddenNode = []
		outputNode = []
		biasNode = []

		#### 1.NODES ADJUSTMENT
		# 1. Nodes for input layer
		# + the input layer bias
		G.add_node(counter, label="bias")    
		pos[counter] = (0, posCounter)
		labels[counter] = "1"
		biasNode.append(counter)
		posCounter += 1
		counter += 1

		# + input nodes
		for i in range(self.input_size):
			G.add_node(counter)
			pos[counter] = (0, posCounter)
			labels[counter] = f"x{i+1}"
			inputNode.append(counter)
			posCounter += 1
			counter += 1

		posCounter = 1
		# 2. Nodes for hidden layer, if exist
		if (self.n_hidden_layer > 0):
			for i in range(self.n_hidden_layer):            
				# + the hidden layer bias
				G.add_node(counter, label="bias")    
				pos[counter] = (1+i, posCounter)
				labels[counter] = "1"
				biasNode.append(counter)
				posCounter += 1
				counter += 1
				# + hidden nodes
				for j in range(self.hidden_layer[i].n_neuron):
					G.add_node(counter)
					pos[counter] = (1+i, posCounter)
					labels[counter] = f"h{i+1}{j+1}"
					hiddenNode.append(counter)
					posCounter += 1
					counter += 1
				plt.annotate(f"{self.hidden_layer[i].activation_function.__name__}", xy=(1+i, posCounter-1), xytext=(1+i, posCounter-1+0.15), ha='center', fontsize=9, fontweight='bold')
							
		posCounter = 1
		# 3. Nodes for output layer
		# + output nodes
		for i in range(len(self.target[0])):
			G.add_node(counter)
			pos[counter] = (1+self.n_hidden_layer, posCounter)
			labels[counter] = f"o{i+1}"
			outputNode.append(counter)
			posCounter += 1
			counter += 1
		plt.annotate(f"{self.output_layer.activation_function.__name__}", xy=(1+self.n_hidden_layer, posCounter-1), xytext=(1+self.n_hidden_layer, posCounter-1+0.15), ha='center', fontsize=9, fontweight='bold')
			
		# nodes style
		options = {"edgecolors": "tab:gray", "node_size": 900, "alpha": 1}
		nx.draw_networkx_nodes(G, pos, nodelist=biasNode, node_color="tab:grey", **options)
		nx.draw_networkx_nodes(G, pos, nodelist=inputNode, node_color="tab:red", **options)
		nx.draw_networkx_nodes(G, pos, nodelist=hiddenNode, node_color="tab:blue", **options)
		nx.draw_networkx_nodes(G, pos, nodelist=outputNode, node_color="tab:green", **options)
			
		#### 2.EDGES ADJUSTMENT
		edgeMap = []
		edge_labels = {}
		# 1. Edges for input layer to output layer (hidden is not exist)
		if (self.n_hidden_layer == 0):
			# connect input to output layer
			for i in range(self.input_size + 1):
				for j in range(len(self.target[0])):
					G.add_edge(i + 1, self.input_size + 2 + j)
					edgeMap.append((i + 1, self.input_size + 2 + j))
					if (i == 0):
						edge_labels[(i + 1, self.input_size + 2 + j)] = f"{self.output_layer.bias[j]:.2f}"
					else:
						edge_labels[(i + 1, self.input_size + 2 + j)] = f"{self.output_layer.weights[i-1][j]:.2f}"
		# 2. Edges for input layer to hidden layer if exist
		elif (self.n_hidden_layer > 0):
			# Connect input to hidden layer
			for i in range(self.input_size + 1):
				for j in range(self.hidden_layer[0].n_neuron):
					G.add_edge(i + 1, self.input_size + 2 + j + 1)
					edgeMap.append((i + 1, self.input_size + 2 + j + 1))
					if (i == 0):
						edge_labels[(i + 1, self.input_size + 2 + j + 1)] = f"{self.hidden_layer[0].bias[j]:.2f}"
					else:
						edge_labels[(i + 1, self.input_size + 2 + j + 1)] = f"{self.hidden_layer[0].weights[i-1][j]:.2f}"
			# Connect hidden layer to other hidden layer
			totalNeuronBeforeThisLayer = 1
			for i in range(self.n_hidden_layer - 1):
				for j in range(self.hidden_layer[i].n_neuron + 1):
					for k in range(self.hidden_layer[i+1].n_neuron):
						G.add_edge(self.input_size + 1 + totalNeuronBeforeThisLayer + j, self.input_size + 2 + totalNeuronBeforeThisLayer + self.hidden_layer[i].n_neuron + k + 1)
						edgeMap.append((self.input_size + 1 + totalNeuronBeforeThisLayer + j, self.input_size + 2 + totalNeuronBeforeThisLayer + self.hidden_layer[i].n_neuron + k + 1))
						if (j == 0):
							edge_labels[(self.input_size + 1 + totalNeuronBeforeThisLayer + j, self.input_size + 2 + totalNeuronBeforeThisLayer + self.hidden_layer[i].n_neuron + k + 1)] = f"{self.hidden_layer[i+1].bias[k]:.2f}"
						else:
							edge_labels[(self.input_size + 1 + totalNeuronBeforeThisLayer + j, self.input_size + 2 + totalNeuronBeforeThisLayer + self.hidden_layer[i].n_neuron + k + 1)] = f"{self.hidden_layer[i+1].weights[j-1][k]:.2f}"
				totalNeuronBeforeThisLayer = totalNeuronBeforeThisLayer + self.hidden_layer[i].n_neuron + 1
			# Connect last hidden layer to output layer
			for i in range(self.hidden_layer[-1].n_neuron + 1):
				for j in range(len(self.target[0])):
					G.add_edge(self.input_size + 1 + totalNeuronBeforeThisLayer + i, self.input_size + 1 + totalNeuronBeforeThisLayer + self.hidden_layer[-1].n_neuron + j + 1)
					edgeMap.append((self.input_size + 1 + totalNeuronBeforeThisLayer + i, self.input_size + 1 + totalNeuronBeforeThisLayer + self.hidden_layer[-1].n_neuron + j + 1))
					if (i == 0):
						edge_labels[(self.input_size + 1 + totalNeuronBeforeThisLayer + i, self.input_size + 1 + totalNeuronBeforeThisLayer + self.hidden_layer[-1].n_neuron + j + 1)] = f"{self.output_layer.bias[j]:.2f}"
					else:
						edge_labels[(self.input_size + 1 + totalNeuronBeforeThisLayer + i, self.input_size + 1 + totalNeuronBeforeThisLayer + self.hidden_layer[-1].n_neuron + j + 1)] = f"{self.output_layer.weights[i-1][j]:.2f}"
									
		# edges style
		edgeOptions = {"width": 2, "alpha": 0.7}
		nx.draw_networkx_edges(G, pos, edgelist=edgeMap, **edgeOptions)
			
		### 3.LABELS ADJUSTMENT
		# labels
		nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight="bold", font_color="whitesmoke")
		nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10, label_pos=0.25, font_weight="bold", font_color="tab:gray")

		plt.tight_layout()
		plt.axis("off")
		plt.show()