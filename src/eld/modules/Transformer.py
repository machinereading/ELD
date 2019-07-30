import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import KoreanUtil
from . import BiContextEmbedding, CNNEmbedding

# entity context emb + relation emb --> transE emb
class Transformer(nn.Module):
	def __init__(self, args):
		super(Transformer, self).__init__()

		self.use_character_embedding = args.use_character_embedding
		self.use_word_context_embedding = args.use_word_context_embedding
		self.use_entity_context_embedding = args.use_entity_context_embedding
		self.use_relation_embedding = args.use_relation_embedding
		assert self.use_character_embedding or self.use_word_context_embedding or self.use_entity_context_embedding or self.use_relation_embedding # use at least one flag

		self.transformer_layer = 1

		self.character_embedding_dim = 50
		self.word_context_embedding_output_dim = 100
		self.entity_context_embedding_output_dim = 100
		self.relation_embedding_output_dim = 100

		self.ffnn_hidden_dim = 100
		self.ffnn_output_dim = 250

		self.ffnn_input_dim = 0
		if self.use_character_embedding:
			self.ffnn_input_dim += self.character_embedding_dim
			self.cv = nn.Embedding(KoreanUtil.jamo_len + len(KoreanUtil.alpha) + 1, args.char_embedding_dim)
			self.character_embedding = nn.Conv1d(1, 1, 1)
		if self.use_word_context_embedding:
			self.ffnn_input_dim += self.word_context_embedding_output_dim
			self.word_context_embedding = BiContextEmbedding("LSTM", 0, self.word_context_embedding_output_dim)
		if self.use_entity_context_embedding:
			self.ffnn_input_dim += self.entity_context_embedding_output_dim
			self.entity_context_embedding = BiContextEmbedding("LSTM", 0, self.entity_context_embedding_output_dim)
		if self.use_relation_embedding:
			self.ffnn_input_dim += self.relation_embedding_output_dim
			self.relation_embedding = CNNEmbedding(1, 1, 1)


		seq = [nn.Linear(self.ffnn_input_dim, self.ffnn_output_dim), nn.Dropout()] if self.transformer_layer < 2 else \
			[nn.Linear(self.ffnn_input_dim, self.ffnn_hidden_dim), nn.Dropout()] + [nn.Linear(self.ffnn_input_dim, self.ffnn_hidden_dim), nn.Dropout()] * (self.transformer_layer - 2) + [nn.Linear(self.ffnn_hidden_dim, self.ffnn_output_dim), nn.Dropout()]
		self.transformer = nn.Sequential(*seq)

	def forward(self, *, character_batch=None, word_context_batch=None, entity_context_batch=None, relation_batch=None):
		assert self.use_character_embedding ^ character_batch
		assert self.use_word_context_embedding ^ word_context_batch
		assert self.use_entity_context_embedding ^ entity_context_batch
		assert self.use_relation_embedding ^ relation_batch

		mid_features = []

		if self.use_character_embedding:
			mid_features.append(self.character_embedding(character_batch))
		if self.use_word_context_embedding:
			mid_features.append(self.word_context_embedding(word_context_batch))
		if self.use_entity_context_embedding:
			mid_features.append(self.entity_context_embedding(entity_context_batch))
		if self.use_relation_embedding:
			mid_features.append(self.relation_embedding(relation_batch))

		ffnn_input = torch.cat(mid_features, dim=-1)

		ffnn_output = self.transformer(ffnn_input)
		return ffnn_output

	def loss(self, pred, label):
		return F.mse_loss(pred, label)

