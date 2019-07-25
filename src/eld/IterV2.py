from ..ds import *
from ..el import EL
from .utils import ELDArgs
import torch

class ELDMain:
	def __init__(self):
		self.args = ELDArgs()

	def train(self):
		pass

	def predict(self, data):
		pass

	def __call__(self, data):
		return self.predict(data)
