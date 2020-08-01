import numpy as np
from evostra.models import FeedForwardNetwork

class CartPoleAgent(FeedForwardNetwork):
	def __init__(self, layer_sizes):
		super().__init__(layer_sizes=layer_sizes)

	def get_action(self, x):
		if self.predict(x) > 0:
			return 1
		else:
			return 0

class MountainCarAgent(FeedForwardNetwork):
	def __init__(self, layer_sizes):
		super().__init__(layer_sizes=layer_sizes)

	def get_action(self, x):
		return np.argmax(self.predict(x))

class PendulumAgent(FeedForwardNetwork):
	def __init__(self, layer_sizes):
		super().__init__(layer_sizes=layer_sizes)

	def get_action(self, x):
		return 1.3*self.predict(x)