import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
	"""
	Transforms ER / EL encoding result into single cluster representation
	Input:
		er: [batch size * cluster size * window size * word embedding dimension]
		el: [batch size * cluster size * window size * entity embedding dimension]
	Output:
		Tensor, [batch size * output dimension size]
	"""
	def __init__(self, args):
		super(Transformer, self).__init__()
		er_input_dim = args.er_output_dim
		el_input_dim = args.el_output_dim
		cluster_size = args.max_cluster_size
		hid_dim = 200
		output_dim = args.transformer_output_dim

		self.transformer = nn.Sequential(
			nn.Conv1d(cluster_size, 1, kernel_size=1),
			nn.MaxPool1d()
			nn.Linear(er_input_dim+el_input_dim, hid_dim),
			nn.ReLU(),
			nn.Linear(hid_dim, output_dim),
			nn.ReLU())

	def forward(self, er, el):
		input_tensor = torch.cat([er, el], dim=0)
		return self.transformer(torch.cat([er, el], dim=0))

	def loss(self, pred, entity_embedding):
		return 