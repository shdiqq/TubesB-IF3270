import numpy as np

# ACTIVATION FUNCTION
def linear(net: float, sum: float = None, derivative: bool = False) -> float:
	"""Fungsi aktivasi linear"""
	if (derivative) :
		return(1)
	else :
		return (net)

def sigmoid(net: float, sum: float = None, derivative: bool = False) -> float:
	"""Fungsi aktivasi sigmoid"""
	if (derivative) :
		return ( (1 / (1 + np.exp(-net)) ) * ( 1 - (1 / (1 + np.exp(-net))) ) )
	else :
		return ( 1 / (1 + np.exp(-net) ) )

def relu(net: float, sum: float = None, derivative: bool = False) -> float:
	"""Fungsi aktivasi ReLU"""
	if (derivative) :
		if ( net  < 0 ) :
			return(0)
		else :
			return(1)
	else :
		return (np.maximum(0, net))

def softmax(net: float, sumNetInEveryNeuron, derivative: bool = False) -> float:
	"""Fungsi aktivasi softmax"""
	sigma = 0
	for i in range (len(sumNetInEveryNeuron)):
		sigma = sigma + np.exp(sumNetInEveryNeuron[i])
	if (derivative) :
		return (np.exp(net) / sigma) * (1 - (np.exp(net) / sigma))
	else :
		return (np.exp(net) / sigma)