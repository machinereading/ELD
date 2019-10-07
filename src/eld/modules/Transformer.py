import torch
import torch.nn as nn
import torch.nn.functional as F

from . import BiContextEncoder, CNNEncoder, SelfAttentionEncoder, FFNNEncoder

# entity context emb + relation emb --> transE emb
class SeparateEncoderBasedTransformer(nn.Module):
	def __init__(self, use_character_embedding, use_word_embedding, use_word_context_embedding, use_entity_context_embedding, use_relation_embedding, use_type_embedding,
	             character_encoder, word_encoder, word_context_encoder, entity_context_encoder, relation_encoder, type_encoder,
	             character_embedding_dim, word_embedding_dim, entity_embedding_dim, relation_embedding_dim, type_embedding_dim,
	             character_encoding_dim, word_encoding_dim, word_context_encoding_dim, entity_context_encoding_dim, relation_encoding_dim, type_encoding_dim,
	             max_jamo, max_word, max_relation):
		super(SeparateEncoderBasedTransformer, self).__init__()
		# legal = encoder_map.keys()
		# for encoder in [character_encoder, word_encoder, entity_encoder, relation_encoder, type_encoder]:
		# 	assert encoder in legal, "Illegal encoder type: %s, must be one of %s" % (encoder, "/".join(legal))
		self.ce_flag = use_character_embedding
		self.we_flag = use_word_embedding
		self.wce_flag = use_word_context_embedding
		self.ee_flag = use_entity_context_embedding
		self.re_flag = use_relation_embedding
		self.te_flag = use_type_embedding
		assert self.ce_flag or self.wce_flag or self.ee_flag or self.re_flag or self.te_flag  # use at least one flag
		self.transformer_layer = 1

		self.transformer_hidden_dim = 100
		self.transformer_output_dim = entity_embedding_dim

		self.transformer_input_dim = 0
		separate_layers = len([x for x in [self.ce_flag, self.we_flag, self.wce_flag, self.ee_flag, self.re_flag, self.te_flag] if x])
		self.ce_dim = self.we_dim = self.wce_dim = self.ee_dim = self.re_dim = self.te_dim = 0
		if self.ce_flag:
			self.character_encoder = CNNEncoder(character_embedding_dim, max_jamo, 2, 3)
			if type(self.character_encoder) is CNNEncoder:
				character_encoding_dim = self.character_encoder.out_size
			self.ce_dim = character_encoding_dim
		if self.we_flag:
			self.word_encoder = CNNEncoder(word_embedding_dim, max_word, 2, 3)
			if type(self.word_encoder) is CNNEncoder:
				word_encoding_dim = self.word_encoder.out_size
			self.we_dim = word_encoding_dim
		if self.wce_flag:
			self.wce_dim = word_context_encoding_dim * 2
			self.word_context_encoder = BiContextEncoder("LSTM", word_embedding_dim, word_context_encoding_dim)
		if self.ee_flag:
			self.ee_dim = entity_context_encoding_dim * 2
			self.entity_context_encoder = BiContextEncoder("LSTM", entity_embedding_dim, entity_context_encoding_dim)
		if self.re_flag:
			self.relation_encoder = CNNEncoder(relation_embedding_dim, max_relation, 2, 2)
			if type(self.relation_encoder) is CNNEncoder:
				relation_encoding_dim = self.relation_encoder.out_size
			self.re_dim = relation_encoding_dim
		if self.te_flag:
			self.type_encoder = FFNNEncoder(type_embedding_dim, type_encoding_dim, (type_embedding_dim + type_encoding_dim) // 2, 2)
			self.te_dim = type_encoding_dim

		# TODO 여러가지 transformer 방식 만들 것
		
		# seq = [nn.Linear(self.transformer_input_dim, self.transformer_output_dim), nn.Dropout()] if self.transformer_layer < 2 else \
		# 	[nn.Linear(self.transformer_input_dim, self.transformer_hidden_dim), nn.Dropout()] + [nn.Linear(self.transformer_input_dim, self.transformer_hidden_dim), nn.Dropout()] * (self.transformer_layer - 2) + [
		# 		nn.Linear(self.transformer_hidden_dim, self.transformer_output_dim),
		# 		nn.Dropout()]
		# self.transformer = nn.Sequential(*seq)

		self.max_input_dim = max(self.ce_dim, self.we_dim, self.wce_dim, self.ee_dim, self.re_dim, self.te_dim)
		self.transformer_input_dim = self.max_input_dim * separate_layers
		self.transformer = SelfAttentionEncoder(self.max_input_dim, 4, 4, separate_layers)


	def forward(self, character_batch, character_len,
	            word_batch, word_len,
	            left_word_context_batch, left_word_context_len,
	            right_word_context_batch, right_word_context_len,
	            left_entity_context_batch, left_entity_context_len,
	            right_entity_context_batch, right_entity_context_len,
	            relation_batch, relation_len,
	            type_batch, type_len):
		mid_features = []

		if self.ce_flag:
			ce = self.character_encoder(character_batch)
			mid_features.append(F.pad(ce, [0, self.max_input_dim - self.ce_dim]))
		if self.we_flag:
			we = self.word_encoder(word_batch)
			mid_features.append(F.pad(we, [0, self.max_input_dim - self.we_dim]))
		if self.wce_flag:
			mid_features.append(F.pad(self.word_context_encoder(left_word_context_batch, left_word_context_len, right_word_context_batch, right_word_context_len), [0, self.max_input_dim - self.wce_dim]))
		if self.ee_flag:
			mid_features.append(F.pad(self.entity_context_encoder(left_entity_context_batch, left_entity_context_len, right_entity_context_batch, right_entity_context_len), [0, self.max_input_dim - self.ee_dim]))
		if self.re_flag:
			mid_features.append(F.pad(self.relation_encoder(relation_batch), [0, self.max_input_dim - self.re_dim]))
		if self.te_flag:
			te = self.type_encoder(type_batch)
			# print(self.type_encoder.out_size, te.size())
			mid_features.append(F.pad(te, [0, self.max_input_dim - self.te_dim]))
		ffnn_input = torch.cat(mid_features, dim=-1)
		ffnn_output = self.transformer(ffnn_input)
		return ffnn_output

	# noinspection PyMethodMayBeStatic
	def loss(self, pred, label):
		return F.mse_loss(pred, label)

class JointTransformer(nn.Module):
	def __init__(self, use_character_embedding, use_word_context_embedding, use_entity_context_embedding, use_relation_embedding, use_type_embedding,
	             character_encoder, word_encoder, entity_encoder, relation_encoder, type_encoder,
	             character_embedding_dim, word_embedding_dim, entity_embedding_dim, relation_embedding_dim, type_embedding_dim,
	             character_encoding_dim, word_encoding_dim, entity_encoding_dim, relation_encoding_dim, type_encoding_dim):
		super(JointTransformer, self).__init__()



	def forward(self, character_batch, character_len,
	            left_word_context_batch, left_word_context_len,
	            right_word_context_batch, right_word_context_len,
	            left_entity_context_batch, left_entity_context_len,
	            right_entity_context_batch, right_entity_context_len,
	            relation_batch, relation_len,
	            type_batch, type_len):
		pass
