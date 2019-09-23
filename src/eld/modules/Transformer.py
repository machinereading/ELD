import torch
import torch.nn as nn
import torch.nn.functional as F

from . import BiContextEncoder, CNNEncoder, SelfAttentionEncoder, encoder_map
from ..utils import ELDArgs

# entity context emb + relation emb --> transE emb
class SeparateEncoderBasedTransformer(nn.Module):
	def __init__(self, use_character_embedding, use_word_context_embedding, use_entity_context_embedding, use_relation_embedding, use_type_embedding,
	             character_encoder, word_encoder, entity_encoder, relation_encoder, type_encoder,
	             character_embedding_dim, word_embedding_dim, entity_embedding_dim, relation_embedding_dim, type_embedding_dim,
	             character_encoding_dim, word_encoding_dim, entity_encoding_dim, relation_encoding_dim, type_encoding_dim):
		super(SeparateEncoderBasedTransformer, self).__init__()
		legal = encoder_map.keys()
		for encoder in [character_encoder, word_encoder, entity_encoder, relation_encoder, type_encoder]:
			assert encoder in legal, "Illegal encoder type: %s, must be one of %s" % (encoder, "/".join(legal))
		self.ce_flag = use_character_embedding
		self.we_flag = use_word_context_embedding
		self.ee_flag = use_entity_context_embedding
		self.re_flag = use_relation_embedding
		self.te_flag = use_type_embedding
		assert self.ce_flag or self.we_flag or self.ee_flag or self.re_flag or self.te_flag  # use at least one flag

		self.transformer_layer = 1

		self.transformer_hidden_dim = 100
		self.transformer_output_dim = entity_embedding_dim

		self.transformer_input_dim = 0
		if self.ce_flag:
			self.transformer_input_dim += character_encoding_dim
			self.character_encoder = nn.Conv1d(1, 1, 1)
		if self.we_flag:
			self.transformer_input_dim += word_encoding_dim
			self.word_context_encoder = BiContextEncoder("LSTM", word_embedding_dim, word_encoding_dim)
		if self.ee_flag:
			self.transformer_input_dim += entity_encoding_dim
			self.entity_context_encoder = BiContextEncoder("LSTM", entity_embedding_dim, entity_encoding_dim)
		if self.re_flag:
			self.transformer_input_dim += relation_encoding_dim
			self.relation_encoder = CNNEncoder(1, 1, 1)
		if self.te_flag:
			self.transformer_input_dim += type_encoding_dim
			self.type_encdoer = SelfAttentionEncoder(512, 8, 8)

		seq = [nn.Linear(self.transformer_input_dim, self.transformer_output_dim), nn.Dropout()] if self.transformer_layer < 2 else \
			[nn.Linear(self.transformer_input_dim, self.transformer_hidden_dim), nn.Dropout()] + [nn.Linear(self.transformer_input_dim, self.transformer_hidden_dim), nn.Dropout()] * (self.transformer_layer - 2) + [
				nn.Linear(self.transformer_hidden_dim, self.transformer_output_dim),
				nn.Dropout()]
		self.transformer = nn.Sequential(*seq)

	def forward(self, character_batch, character_len,
	            left_word_context_batch, left_word_context_len,
	            right_word_context_batch, right_word_context_len,
	            left_entity_context_batch, left_entity_context_len,
	            right_entity_context_batch, right_entity_context_len,
	            relation_batch, relation_len,
	            type_batch, type_len):
		mid_features = []

		if self.ce_flag:
			mid_features.append(self.character_encoder(character_batch))
		if self.we_flag:
			mid_features.append(self.word_context_encoder(left_word_context_batch, right_word_context_batch))
		if self.ee_flag:
			mid_features.append(self.entity_context_encoder(left_entity_context_batch, right_entity_context_batch))
		if self.re_flag:
			mid_features.append(self.relation_encoder(relation_batch))
		if self.te_flag:
			mid_features.append(self.type_encoder(type_batch))

		ffnn_input = torch.cat(mid_features, dim=-1)

		ffnn_output = self.transformer(ffnn_input)
		return ffnn_output

	# noinspection PyMethodMayBeStatic
	def loss(self, pred, label):
		return F.mse_loss(pred, label)

class JointTransformer(nn.Module):
	def __init__(self, args: ELDArgs):
		super(JointTransformer, self).__init__()

	def forward(self, *input):
		pass
