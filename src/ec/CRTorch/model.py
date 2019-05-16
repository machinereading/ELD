import torch
import torch.nn as nn
import torch.nn.functional as F

from .Attention import Attention
from ..utils import ECArgs
from ...utils import Embedding

class CorefModel(nn.Module):
	def __init__(self, args: ECArgs):
		super(CorefModel, self).__init__()
		self.attn_size = args.attn_dim
		self.target_device = args.device
		self.max_precedents = args.max_precedent
		self.word_embedding = Embedding.load_embedding(args.embedding_type, args.embedding_path).to(self.target_device)
		self.use_dropout = args.use_dropout
		if self.use_dropout:
			self.dropout = nn.Dropout()
		self.attention = Attention(self.attn_size).to(self.target_device)
		self.lstm_encoder = nn.LSTM(self.word_embedding.embedding_dim, args.lstm_dim, batch_first=True).to(self.target_device)
		self.mention_scorer = nn.Sequential(
				nn.Linear(self.attn_size, args.hidden_dim),
				nn.ReLU(),
				nn.Linear(args.hidden_dim, args.hidden_dim),
				nn.ReLU()
		).to(self.target_device)
		self.wm = nn.Parameter(torch.randn(args.hidden_dim, requires_grad=True)).to(self.target_device)
		self.wa = nn.Parameter(torch.randn(args.hidden_dim, requires_grad=True)).to(self.target_device)
		self.antecedent_scorer = nn.Sequential(
				nn.Linear(self.attn_size * 3, args.hidden_dim),
				nn.ReLU(),
				nn.Linear(args.hidden_dim, args.hidden_dim),
				nn.ReLU()
		).to(self.target_device)

	def forward(self, word_seq, mask):
		"""

		:param word_seq: Tensor, rank 2, batch size * max vocab size in seq
		:param mask: List, batch size * [(start, end)]
		:return:
		"""
		batch_size = word_seq.size()[0]
		word_size = word_seq.size()[1]
		we = self.word_embedding(word_seq)
		if self.use_dropout:
			we = self.dropout(we)
		# print(we.size())
		span_representation, (h, c) = self.lstm_encoder(we)
		span_representation.view(batch_size, word_size, -1)

		h = h.transpose(0, 1)
		targets = self.get_target_mentions(span_representation, h, mask)  # batch * max word size * attention size

		mention_score = self.mention_scorer(targets)
		mention_score = self.batch_dot_product(mention_score, self.wm)
		# print(mention_score.size()) # batch * max word size
		pairs, indicator = self.pair_generator(targets)
		pair_score = self.antecedent_scorer(pairs)
		pair_score = self.batch_dot_product(pair_score.transpose(0, 1), self.wa)
		pair_score = pair_score.transpose(0, 1)

		final_score = self.final_score(mention_score, pair_score, indicator)
		return final_score

	def batch_dot_product(self, score, mat):
		_, ss1, ss2 = score.size()
		mat = mat.repeat(ss1, 1)
		return torch.stack([torch.bmm(i.view(ss1, 1, ss2), mat.view(ss1, ss2, 1)) for i in score]).squeeze()

	def get_target_mentions(self, vec, ctx, mask):
		tensors = []
		max_items = max([len(x) for x in mask])
		for i, m in enumerate(mask):
			x = []
			for s, e in m:
				if s == e == -1:
					continue
				# attn = self.attention(vec[i, s:e, :], ctx[i, s:e, :])[0]
				sum = torch.sum(vec[i, s:e, :], dim=0)
				# print(s, e, vec[i, s:e, :].size(), sum.size())
				x.append(sum)
			x += [torch.zeros(self.attn_size).to(self.target_device)] * (max_items - len(x))
			tensors.append(torch.stack(x))

		return torch.stack(tensors).to(self.target_device)

	def pair_generator(self, vec):
		result = []
		pair_indicator = []
		for i in range(vec.size()[1]):
			for j in range(max(i - self.max_precedents, 0), i):
				jvec = vec[:, j, :]
				ivec = vec[:, i, :]
				result.append(torch.cat((jvec, ivec, jvec * ivec), dim=-1))
				pair_indicator.append((j, i))
		# max voca ^ 2 * batch size * (3*attn)
		# result = torch.tensor(result).to(self.target_device).transpose(0, 1)
		return torch.stack(result), pair_indicator

	def final_score(self, mention_score, pair_score, pair_indicator):
		result = []  #
		batch_size = mention_score.size()[0]
		mention_score = mention_score.transpose(0, 1)
		indicator = []
		precedents = []
		for i, ms in enumerate(mention_score):  # mention * batch
			precedent = [torch.tensor([-float("Inf")] * batch_size).to(self.target_device) for _ in range(self.max_precedents)]
			idx = 0
			for score, (s, e) in zip(pair_score, pair_indicator):
				if e != i: continue
				precedent[idx] = mention_score[i] + mention_score[s] + score

				idx += 1
			precedent.append(torch.zeros(batch_size).to(self.target_device))
			precedent = torch.stack(precedent)
			precedents.append(precedent)
		# zero = torch.zeros(ms.size()).to(self.target_device)  # 모든 batch에 대한 i, -1의 score
		# result.append(zero)
		# indicator.append((-1, i))
		# for score, (s, e) in zip(pair_score, pair_indicator):
		# 	if e != i: continue
		# 	result.append(mention_score[i] + mention_score[s] + score)
		# 	indicator.append((s, e))
		precedents = torch.stack(precedents).to(self.target_device)  # mention * max_precedent * batch
		precedents = precedents.transpose(1, 2)
		precedents = precedents.transpose(0, 1)  # batch * mention * max_precedent
		result = F.softmax(precedents)
		return result

	# scores = torch.stack(scores).to(self.target_device).transpose(0, 1)  # [0, 0], [a, b], [c, d] -> [0, a, c], [0, b, d] -> batch * ith mention coreference score
	#
	# scores = F.sigmoid(scores)
	# result.append(scores)
	# result = torch.stack(result).to(self.target_device)
	# result = F.sigmoid(result)
	# return result, indicator

	def loss(self, prediction, indicator, label):
		real_labels = self.get_real_labels(indicator, label)
		return F.binary_cross_entropy(prediction, real_labels)

	def loss_v2(self, precedent, precedent_label):
		precedent = precedent.view(-1, self.max_precedents + 1)
		precedent_label = precedent_label.view(-1, self.max_precedents + 1)
		return F.multilabel_margin_loss(precedent, precedent_label)

	def get_real_labels(self, indicator, label):
		label = label.transpose(0, 1)
		has_same_cluster_in_precedent = []
		real_labels = []
		for i, x in enumerate(label):
			if i == 0:
				has_same_cluster_in_precedent.append(torch.zeros(label.size()[1]).to(self.target_device))
				continue
			precedent = label[i - self.max_precedents:i, :]
			cluster = x.repeat(max(1, precedent.size()[0]), 1)

			eq = torch.eq(precedent, cluster).transpose(0, 1)
			has_same_cluster_in_precedent.append(torch.tensor([1. if any([x == 1 for x in t]) else 0. for t in eq]).to(self.target_device))
		for i, j in indicator:
			if i == -1:
				real_labels.append(has_same_cluster_in_precedent[j])
				continue
			target_cluster = label[i, :]
			cluster = label[j, :]
			same = torch.eq(target_cluster, cluster).float()
			real_labels.append(same)

		real_labels = torch.stack(real_labels).to(self.target_device)
		return real_labels
