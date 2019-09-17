import torch
import torch.nn as nn
import torch.nn.functional as F

from src.eld.utils import ELDArgs
from src.utils import KoreanUtil
from . import BiContextEmbedding, CNNEmbedding

# entity context emb + relation emb --> transE emb
class Transformer(nn.Module):
	def __init__(self, args: ELDArgs):
		super(Transformer, self).__init__()

		self.ce_flag = args.use_character_embedding
		self.we_flag = args.use_word_context_embedding
		self.ee_flag = args.use_entity_context_embedding
		self.re_flag = args.use_relation_embedding
		self.te_flag = args.use_type_embedding
		assert self.ce_flag or self.we_flag or self.ee_flag or self.re_flag or self.te_flag  # use at least one flag

		self.transformer_layer = 1

		self.character_embedding_dim = 50
		self.word_context_embedding_output_dim = 100
		self.entity_context_embedding_output_dim = 100
		self.relation_embedding_output_dim = 100

		self.transformer_hidden_dim = 100
		self.transformer_output_dim = 250

		self.transformer_input_dim = 0
		if self.ce_flag:
			self.transformer_input_dim += self.character_embedding_dim
			self.cv = nn.Embedding(KoreanUtil.jamo_len + len(KoreanUtil.alpha) + 1, args.ce_dim)
			self.character_embedding = nn.Conv1d(1, 1, 1)
		if self.we_flag:
			self.transformer_input_dim += self.word_context_embedding_output_dim
			self.word_context_embedding = BiContextEmbedding("LSTM", 0, self.word_context_embedding_output_dim)
		if self.ee_flag:
			self.transformer_input_dim += self.entity_context_embedding_output_dim
			self.entity_context_embedding = BiContextEmbedding("LSTM", 0, self.entity_context_embedding_output_dim)
		if self.re_flag:
			self.transformer_input_dim += self.relation_embedding_output_dim
			self.relation_embedding = CNNEmbedding(1, 1, 1)

		seq = [nn.Linear(self.transformer_input_dim, self.transformer_output_dim), nn.Dropout()] if self.transformer_layer < 2 else \
			[nn.Linear(self.transformer_input_dim, self.transformer_hidden_dim), nn.Dropout()] + [nn.Linear(self.transformer_input_dim, self.transformer_hidden_dim), nn.Dropout()] * (self.transformer_layer - 2) + [nn.Linear(self.transformer_hidden_dim, self.transformer_output_dim),
		                                                                                                                                                                                                    nn.Dropout()]
		self.transformer = nn.Sequential(*seq)

	def forward(self, *, character_batch=None, word_context_batch=None, entity_context_batch=None, relation_batch=None, type_batch=None):
		# flag and batch match
		assert not (self.ce_flag ^ character_batch)
		assert not (self.we_flag ^ word_context_batch)
		assert not (self.ee_flag ^ entity_context_batch)
		assert not (self.re_flag ^ relation_batch)
		assert not (self.te_flag ^ type_batch)
		mid_features = []

		if self.ce_flag:
			mid_features.append(self.character_embedding(character_batch))
		if self.we_flag:
			mid_features.append(self.word_context_embedding(word_context_batch))
		if self.ee_flag:
			mid_features.append(self.entity_context_embedding(entity_context_batch))
		if self.re_flag:
			mid_features.append(self.relation_embedding(relation_batch))

		ffnn_input = torch.cat(mid_features, dim=-1)

		ffnn_output = self.transformer(ffnn_input)
		return ffnn_output

	def loss(self, pred, label):
		return F.mse_loss(pred, label)
