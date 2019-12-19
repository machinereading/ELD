import torch
import torch.nn as nn
import torch.nn.functional as F

from ....utils import TensorUtil

def get_transformer(args):
	if args.transformer == "nn":
		return NNTransformer(args)
	elif args.transformer == "avg":
		return AvgTransformer(args)

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
		cluster_size = args.chunk_size
		self.hid_dim = 100
		self.channels = args.channel
		self.test_transformer = nn.Sequential(
				nn.Conv1d(cluster_size, self.channels, kernel_size=1),
				# n=batch, input_channel = cluster_size, l = embedding dimension
				nn.MaxPool1d(2)
		)
		self.word_transformer = nn.Sequential(
				nn.Conv1d(cluster_size, self.channels, kernel_size=1),
				# n=batch, input_channel = cluster_size, l = embedding dimension
				nn.MaxPool1d(2),
				nn.Linear(er_input_dim, self.hid_dim),
				nn.ReLU()
		)
		self.entity_transformer = nn.Sequential(
				nn.Conv1d(cluster_size, self.channels, kernel_size=1),
				nn.MaxPool1d(2),
				nn.Linear(el_input_dim, self.hid_dim),
				nn.ReLU()
		)
		self.final_transformer = nn.Linear(args.jamo_embed_dim + 2 * self.channels * self.hid_dim, args.transform_dim)

	def forward(self, jamo, word, entity):
		# batch * cluster size * embedding size
		# print(jamo.size(), word.size(), entity.size())
		# print(self.test_transformer(word).size())

		j = TensorUtil.nonzero_avg_stack(jamo)
		w = self.word_transformer(word).view(-1, self.channels * self.hid_dim)
		e = self.entity_transformer(entity).view(-1, self.channels * self.hid_dim)
		return self.final_transformer(torch.cat((j, w, e), dim=-1))

	def loss(self, pred, entity_embedding):
		return

class AvgTransformer(nn.Module):
	def __init__(self, args):
		super(AvgTransformer, self).__init__()
		self.use_stddev = getattr(args, "use_stddev", False)
		er_input_dim = args.er_output_dim
		el_input_dim = args.el_output_dim
		in_dim = args.jamo_embed_dim + er_input_dim * 2 + el_input_dim * 2
		if args.use_stddev:
			in_dim += 2
		self.transformer = nn.Linear(in_dim, args.transform_dim)  # 48 is hard coded

	def forward(self, jamo, word, entity):
		# input: batch size * max voca * embedding size
		# print(jamo.size(), word.size(), entity.size())
		j = TensorUtil.nonzero_avg_stack(jamo)
		w = TensorUtil.nonzero_avg_stack(word)
		e = TensorUtil.nonzero_avg_stack(entity)
		if self.use_stddev:
			w_std = TensorUtil.nonzero_std_dev(word).view(-1, 1)
			e_std = TensorUtil.nonzero_std_dev(entity).view(-1, 1)
			return F.relu(self.transformer(torch.cat((j, w, w_std, e, e_std), -1)))
		return F.relu(self.transformer(torch.cat((j, w, e), -1)))
	# print(w_std.size(), e_std.size())
	# print(j.size(), w.size(), e.size())

class Identity(nn.Module):
	def __init__(self):
		super(Identity, self).__init__()

	def forward(self, in_tensor):
		return in_tensor

# def load_transformer(mode, cluster_size, er_input_dim, el_input_dim):
# 	if mode == "cnn":
# 		er_transformer = nn.Sequential(
# 				nn.Conv1d(cluster_size, 3, kernel_size=1),
# 				# n=batch, input_channel = cluster_size, l = embedding dimension
# 				nn.MaxPool1d(2),
# 				nn.Linear(er_input_dim // 2, hid_dim),
# 				nn.ReLU(),
# 				nn.Linear(hid_dim, output_dim),
# 				nn.ReLU()
# 		)
# 		el_transformer = nn.Sequential(
# 				nn.Conv1d(cluster_size, 3, kernel_size=1),
# 				nn.MaxPool1d(2),
# 				nn.Linear(el_input_dim // 2, hid_dim),
# 				nn.ReLU(),
# 				nn.Linear(hid_dim, output_dim),
# 				nn.ReLU()
# 		)
# 	elif mode == "avg":  # 어차피 cosine distance 쓸거면 zero를 무시하던 말던 상관없음
# 		er_transformer = Identity()
# 		el_transformer = Identity()
#
# 	return er_transformer, el_transformer
