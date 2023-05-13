import os
import sys

script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, '..', 'class')
sys.path.append(mymodule_dir)

import json
import numpy as np
from mbgd import MiniBatchGradientDescent

def generate_model(filePath):
	model = open(filePath)
	model_data = json.loads(model.read())
	model.close()

	#set variable
	activation_function = []
	n_neuron = []
	input_size = None
	input_data = None
	bias = []
	weights = []
	target = None
	learning_rate = None
	batch_size = None
	max_iteration = None
	error_threshold = None
	stopped_by = None
	bias_final = []
	weights_final = []
    
	for x in model_data: # case dan expect
		for y in model_data[str(x)] : # model, input, initial_weights, target, learning_parameters
			if (str(y) == "model") :
				for z in (model_data[str(x)])[str(y)] : #input_size, layers
					if (str(z) == "layers" ) : # array of list
						for j in ((model_data[str(x)])[str(y)])[str(z)] :
							activation_function.append(j['activation_function'])
							n_neuron.append(j['number_of_neurons'])
					else : # "input_size"
						input_size = ((model_data[str(x)])[str(y)])[str(z)]
			elif (str(y) == 'input'):
					input_data = (model_data[str(x)])[str(y)]
			elif (str(y) == 'initial_weights'):
				for z in range(len(((model_data[str(x)])[str(y)]))) :
					bias.append(model_data[str(x)][str(y)][z][0])
					weights.append(model_data[str(x)][str(y)][z][1:])
			elif (str(y) == 'target'):
				target = ((model_data[str(x)])[str(y)])
			elif (str(y) == 'learning_parameters'):
				for z in (model_data[str(x)])[str(y)] : #learning_rate, batch_size, max_iteration, error_tresh
					if ( str(z) == "learning_rate") :
						learning_rate = ((model_data[str(x)])[str(y)])[str(z)]
					elif ( str(z) == "batch_size") :
						batch_size = ((model_data[str(x)])[str(y)])[str(z)]
					elif ( str(z) == "max_iteration") :
						max_iteration = ((model_data[str(x)])[str(y)])[str(z)]
					else : #error_threshold
						error_threshold = ((model_data[str(x)])[str(y)])[str(z)]
			elif (str(y) == 'stopped_by'):
				stopped_by = ((model_data[str(x)])[str(y)])
			elif (str(y) == 'final_weights'):
				for z in range(len(((model_data[str(x)])[str(y)]))) :
					bias_final.append(model_data[str(x)][str(y)][z][0])
					weights_final.append(model_data[str(x)][str(y)][z][1:])

	#create MiniBatchGradientDescent
	mbgd = MiniBatchGradientDescent(max_iter=int(max_iteration), batch_size=int(batch_size), error_threshold=float(error_threshold), learning_rate=float(learning_rate))

	#Add input, bias, and weights to layer
	# Input Layer
	mbgd.add_layer_node('input_layer', int(input_size), None, 'None', (input_data), None, None)
	# Hidden & Output Layer
	for i in range(len(bias)) :
		typeLayer = ""
		if (i != (len(bias)-1)):
			typeLayer = "hidden_layer"
		else:
			typeLayer = "output_layer"
		mbgd.add_layer_node(typeLayer, None, int(n_neuron[i]), str(activation_function[i]), None, weights[i], bias[i])

	# Add target
	mbgd.set_target(target)

	# Add stopped_by
	mbgd.set_stopped_by(stopped_by)

	# Add bias_final
	mbgd.set_bias_final(bias_final)

	# Add weights_final
	mbgd.set_weights_final(weights_final)
    
	return mbgd