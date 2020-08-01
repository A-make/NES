import numpy as np
from utils import relu, sigmoid

def nn_forward(weights, a):
	for w, b in weights[:-1]:
		z = np.dot(w,a) + b
		a = relu(z)
	w, b = weights[-1]
	return np.dot(w,a) + b

class BaseAgent(object):
	def __init__(self, layer_sizes):
		self.layer_sizes = layer_sizes
		self.params_length = 0
		for m,n in zip(layer_sizes[:-1], layer_sizes[1:]):
			self.params_length += m*n + n
		self.weights = None

	def set_weights(self, params):
		self.weights = []
		i = 0
		for m,n in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
			p = m*n
			self.weights.append( (params[i:i+p].reshape(n,m), params[i+p:i+p+n].reshape(n,)) )
			i += p+n

class CartPoleAgent(BaseAgent):
	def __init__(self, layer_sizes):
		super().__init__(layer_sizes)
	
	def get_action(self, x):
		if nn_forward(self.weights, x) > 0:
			return 1
		else:
			return 0

class MountainCarAgent(BaseAgent):
	def __init__(self, layer_sizes):
		super().__init__(layer_sizes)

	def get_action(self, x):
		return np.argmax(nn_forward(self.weights, x))

class PendulumAgent(BaseAgent):
	def __init__(self, layer_sizes):
		super().__init__(layer_sizes)

	def get_action(self, x):
		return 2*np.tanh(nn_forward(self.weights, x))