import torch
import torch.nn as nn
import torch.nn.functional as F
from ....utils import TimeUtil

@TimeUtil.measure_time
def nonzero_avg_stack(tensor):
	# input: 3 dimension tensor - max voca * max jamo * embedding size
	# output: 2 dimension tensor - max voca * embedding size
	# 지금 이거 원하는대로 동작 안함. 지금 임베딩이 빠지는 중이니까 확인 필요
	avg = []
	nz = nonzero_item_count(tensor)
	tensor = tensor.sum(1) / nz if nz > 0 else tensor
	return tensor


def nonzero_item_count(tensor):
	# input: 2 dimension tensor
	# output: single tensor
	result = 0
	for vec in tensor:
		if torch.sum(vec) == 0:
			continue
		result += 1
	return result

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
		er_input_dim = args.er_output_dim
		el_input_dim = args.el_output_dim
		jamo_dim = args.char_embedding_dim
		self.transformer = nn.Linear(48+er_input_dim*2+el_input_dim*2, args.transform_dim)

	def forward(self, jamo, word, entity):
		# input: batch size * max voca * embedding size
		# print(jamo.size(), word.size(), entity.size())
		j = nonzero_avg_stack(jamo)
		w = nonzero_avg_stack(word)
		e = nonzero_avg_stack(entity)
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