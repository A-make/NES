import numpy as np
from torch import FloatTensor, argmax
import torch.nn as nn
# import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

class Net(nn.Module):
	def __init__(self, layer_sizes):
		super().__init__()
		#self.memory = np.zeros(20)
		topology = []
		for m,n in zip(layer_sizes[:-2], layer_sizes[1:-1]):
			 topology.append(nn.Linear(m,n))
			 topology.append(nn.ReLU())
		m,n = layer_sizes[-2],layer_sizes[-1]
		topology.append(nn.Linear(m,n))
		self.net = nn.Sequential(*topology)
		# Remove grad from model
		for param in self.net.parameters():
			param.requires_grad = False
	
	def get_weights(self):
		'''return parameters (weights) as 1D vector. Need to convert to numpy?'''
		return parameters_to_vector(self.net.parameters())
	
	def set_weights(self, weights):
		'''set parameters from 1D vector'''
		vector_to_parameters(weights, self.net.parameters())

class CartPoleAgent(Net):
	def __init__(self, layer_sizes, swing_up = False):
		super().__init__(layer_sizes)
		self.swing_up = swing_up

	def get_action(self, obs):
		obs = FloatTensor(obs)
		if self.net(obs) > 0: 
			return 1
		else:
			return 0

class MountainCarAgent(Net):
	def __init__(self, layer_sizes):
		super().__init__(layer_sizes)
	
	def get_action(self, obs):
		obs = FloatTensor(obs)
		return argmax(self.net(obs)).item()

class AcrobotAgent(Net):
	def __init__(self, layer_sizes):
		super().__init__(layer_sizes)
	
	def get_action(self, obs):
		obs = FloatTensor(obs)
		return np.argmax(self.net(obs))