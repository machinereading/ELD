import torch
import torch.nn as nn
import torch.nn.functional as F
from ....utils import TimeUtil, TensorUtil



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
		hid_dim = 200
		output_dim = args.transform_dim
		self.word_transformer = nn.Sequential(
			nn.Conv1d(cluster_size, 3, kernel_size=1), # n=batch, input_channel = cluster_size, l = embedding dimension
			nn.MaxPool1d(2),
			nn.Linear(er_input_dim // 2, hid_dim),
			nn.ReLU(),
			nn.Linear(hid_dim, output_dim),
			nn.ReLU()
		)
		self.entity_transformer = nn.Sequential(
			nn.Conv1d(cluster_size, 3, kernel_size=1),
			nn.MaxPool1d(2),
			nn.Linear(el_input_dim // 2, hid_dim),
			nn.ReLU(),
			nn.Linear(hid_dim, output_dim),
			nn.ReLU()
		)


	def forward(self, jamo, word, entity):
		# batch * cluster size * embedding size
		return self.word_transformer(word), self.entity_transformer(entity)

	def loss(self, pred, entity_embedding):
		return

class AvgTransformer(nn.Module):
	def __init__(self, args):
		super(AvgTransformer, self).__init__()
		er_input_dim = args.er_output_dim
		el_input_dim = args.el_output_dim
		jamo_dim = args.char_embedding_dim
		self.transformer = nn.Linear(48+er_input_dim*2+el_input_dim*2, args.transform_dim) # 48 is hard coded

	def forward(self, jamo, word, entity):
		# input: batch size * max voca * embedding size
		# print(jamo.size(), word.size(), entity.size())
		j = TensorUtil.nonzero_avg_stack(jamo)
		w = TensorUtil.nonzero_avg_stack(word)
		e = TensorUtil.nonzero_avg_stack(entity)
		# print(j.size(), w.size(), e.size())
		return F.relu(self.transformer(torch.cat([j, w, e], -1)))
		



	


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