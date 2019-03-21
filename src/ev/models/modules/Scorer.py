import torch
import torch.nn as nn

from . import ModelFactory

class ERScorer(nn.Module):
	def __init__(self, args, input_dim):
		super(ERScorer, self).__init__()
		self.model = ModelFactory.load_model(args.er_model, input_dim, args.er_output_dim)


	def forward(self, word_embeddings):
		return self.model(word_embeddings)

class ELScorer(nn.Module):
	def __init__(self, args, input_dim):
		super(ELScorer, self).__init__(args.el_model, input_dim, args.el_output_dim)
		pass

	def forward(self, entity_embeddings):
		pass

class ECScorer(nn.Module):
	def __init__(self, args):
		super(ECScorer, self).__init__(args.ec_model)
		pass

	def forward(self):
		pass
