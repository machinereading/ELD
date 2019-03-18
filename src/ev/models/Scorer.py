import torch
import torch.nn as nn

from . import ModelFactory

class ERScorer(nn.Module):
	def __init__(self, args):
		super(ERScorer, self).__init__()
		self.model = ModelFactory.load_model(args.er_model)


	def forward(self):
		pass

class ELScorer(nn.Module):
	def __init__(self, args):
		super(ELScorer, self).__init__(args.el_model)
		pass

	def forward(self):
		pass

class ECScorer(nn.Module):
	def __init__(self, args):
		super(ECScorer, self).__init__(args.ec_model)
		pass

	def forward(self):
		pass
