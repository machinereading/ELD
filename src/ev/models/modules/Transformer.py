import torch
import torch.nn as nn
import torch.nn.functional as F

class NNTransformer(nn.Module):
	"""
	Transforms ER / EL encoding result into single cluster representation
	Input:
		er: [batch size * cluster size * window size * word embedding dimension]
		el: [batch size * cluster size * window size * entity embedding dimension]
	Output:
		Tensor, [batch size * output dimension size]
	"""
	def __init__(self, args):
		super(NNTransformer, self).__init__()
		er_input_dim = args.er_output_dim
		el_input_dim = args.el_output_dim
		cluster_size = args.max_cluster_size
		hid_dim = 200
		output_dim = args.transformer_output_dim
		if args.transformer_model == "nn":
			self.er_transformer = nn.Sequential(
				nn.Conv1d(cluster_size, 3, kernel_size=1), # n=batch, input_channel = cluster_size, l = embedding dimension
				nn.MaxPool1d(2),
				nn.Linear(er_input_dim // 2, hid_dim),
				nn.ReLU(),
				nn.Linear(hid_dim, output_dim),
				nn.ReLU()
			)
			self.el_transformer = nn.Sequential(
				nn.Conv1d(cluster_size, 3, kernel_size=1),
				nn.MaxPool1d(2),
				nn.Linear(el_input_dim // 2, hid_dim),
				nn.ReLU(),
				nn.Linear(hid_dim, output_dim),
				nn.ReLU()
			)
		elif args.transformer_model == "avg":
			self.er_transformer

	def forward(self, er, el):
		# batch * cluster size * embedding size
		return self.er_transformer(er), self.el_transformer(el)

	def loss(self, pred, entity_embedding):
		return

class AvgTransformer(nn.Module):
	def __init__(self, args):
		super(AvgTransformer, self).__init__()

	def forward(self, er, el):
		return er.mean(1), el.mean(1)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, in_tensor):
        return in_tensor

def load_transformer(mode, cluster_size, er_input_dim, el_input_dim):
	if mode == "cnn":
		er_transformer = nn.Sequential(
			nn.Conv1d(cluster_size, 3, kernel_size=1), # n=batch, input_channel = cluster_size, l = embedding dimension
			nn.MaxPool1d(2),
			nn.Linear(er_input_dim // 2, hid_dim),
			nn.ReLU(),
			nn.Linear(hid_dim, output_dim),
			nn.ReLU()
		)
		el_transformer = nn.Sequential(
			nn.Conv1d(cluster_size, 3, kernel_size=1),
			nn.MaxPool1d(2),
			nn.Linear(el_input_dim // 2, hid_dim),
			nn.ReLU(),
			nn.Linear(hid_dim, output_dim),
			nn.ReLU()
		)
	elif mode == "avg": # 어차피 cosine distance 쓸거면 zero를 무시하던 말던 상관없음
		er_transformer = Identity()
		el_transformer = Identity()

	return er_transformer, el_transformer