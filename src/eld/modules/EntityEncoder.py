import torch
import torch.nn as nn
import torch.nn.functional as F

from . import BiContextEncoder, CNNEncoder, FFNNEncoder, SelfAttentionEncoder

# entity context emb + relation emb --> transE emb
class SeparateEntityEncoder(nn.Module):
	def __init__(self, use_character_embedding, use_word_embedding, use_word_context_embedding, use_entity_context_embedding, use_relation_embedding, use_type_embedding,
	             character_encoder, word_encoder, word_context_encoder, entity_context_encoder, relation_encoder, type_encoder,
	             character_embedding_dim, word_embedding_dim, entity_embedding_dim, relation_embedding_dim, type_embedding_dim,
	             character_encoding_dim, word_encoding_dim, word_context_encoding_dim, entity_context_encoding_dim, relation_encoding_dim, type_encoding_dim,
	             max_jamo, max_word, max_relation):
		super(SeparateEntityEncoder, self).__init__()
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
		self.transformer_layer = 2

		self.transformer_hidden_dim = 300
		self.transformer_output_dim = entity_embedding_dim

		self.transformer_input_dim = 0
		separate_layers = len([x for x in [self.ce_flag, self.we_flag, self.wce_flag, self.ee_flag, self.re_flag, self.te_flag] if x])
		self.ce_dim = self.we_dim = self.wce_dim = self.ee_dim = self.re_dim = self.te_dim = 0
		# print(character_embedding_dim, word_embedding_dim, entity_embedding_dim, relation_embedding_dim, type_embedding_dim)
		if self.ce_flag:
			if character_encoder.lower() == "cnn":
				self.character_encoder = CNNEncoder(character_embedding_dim, max_jamo, 2, 3)
				character_encoding_dim = self.character_encoder.out_size
				self.ce_dim = character_encoding_dim
			elif character_encoder.lower() == "selfattn":
				self.character_encoder = SelfAttentionEncoder(character_embedding_dim, 6, 5, max_jamo, output_dim=character_encoding_dim)
				self.ce_dim = character_encoding_dim
		if self.we_flag:
			if word_encoder.lower() == "cnn":
				self.word_encoder = CNNEncoder(word_embedding_dim, max_word, 2, 3)
				word_encoding_dim = self.word_encoder.out_size
				self.we_dim = word_encoding_dim
			elif word_encoder.lower() == "selfattn":
				self.word_encoder = SelfAttentionEncoder(word_embedding_dim, 6, 5, max_word, output_dim=word_encoding_dim)
				self.we_dim = word_encoding_dim
		if self.wce_flag:
			if word_context_encoder.lower() == "bilstm":
				self.wce_dim = word_context_encoding_dim * 2
				self.word_context_encoder = BiContextEncoder("LSTM", word_embedding_dim, word_context_encoding_dim)
		if self.ee_flag:
			if entity_context_encoder.lower() == "bilstm":
				self.ee_dim = entity_context_encoding_dim * 2
				self.entity_context_encoder = BiContextEncoder("LSTM", entity_embedding_dim, entity_context_encoding_dim)
		if self.re_flag:
			if relation_encoder.lower() == "cnn":
				self.relation_encoder = CNNEncoder(relation_embedding_dim, max_relation, 2, 2)
				relation_encoding_dim = self.relation_encoder.out_size
				self.re_dim = relation_encoding_dim
			# elif relation_encoder.lower() == "selfattn":
			# 	self.relation_encoder = SelfAttentionEncoder(relation_embedding_dim, 4, 5, 1, output_dim=relation_encoding_dim)
			# 	self.re_dim = relation_encoding_dim
		if self.te_flag:
			if type_encoder.lower() == "ffnn":
				self.type_encoder = FFNNEncoder(type_embedding_dim, type_encoding_dim, (type_embedding_dim + type_encoding_dim) // 2, 2)
				self.te_dim = type_encoding_dim
			elif type_encoder.lower() == "selfattn":
				self.type_encoder = SelfAttentionEncoder(type_embedding_dim, 6, 5, 1, output_dim=type_encoding_dim)
				self.te_dim = type_encoding_dim

		self.max_input_dim = max(self.ce_dim, self.we_dim, self.wce_dim, self.ee_dim, self.re_dim, self.te_dim)
		self.transformer_input_dim = self.max_input_dim * separate_layers

		seq = [nn.Linear(self.transformer_input_dim, self.transformer_output_dim), nn.Dropout()] if self.transformer_layer < 2 else \
			[nn.Linear(self.transformer_input_dim, self.transformer_hidden_dim), nn.Dropout()] + [nn.Linear(self.transformer_input_dim, self.transformer_hidden_dim), nn.Dropout()] * (self.transformer_layer - 2) + [
				nn.Linear(self.transformer_hidden_dim, self.transformer_output_dim),
				nn.Dropout()]
		self.transformer = nn.Sequential(*seq)

		# self.transformer = SelfAttentionEncoder(self.max_input_dim, 4, 4, separate_layers)

		self.encoder_output = None
		binary_encoder_seq = [nn.Linear(self.transformer_input_dim, self.transformer_hidden_dim), nn.Dropout()] + [nn.Linear(self.transformer_input_dim, self.transformer_hidden_dim), nn.Dropout()] * (self.transformer_layer - 2) + [
			nn.Linear(self.transformer_hidden_dim, 1), nn.Dropout()]
		self.binary_encoder = nn.Sequential(*binary_encoder_seq)

	def forward(self, character_batch, character_len,
	            word_batch, word_len,
	            left_word_context_batch, left_word_context_len,
	            right_word_context_batch, right_word_context_len,
	            left_entity_context_batch, left_entity_context_len,
	            right_entity_context_batch, right_entity_context_len,
	            relation_batch, relation_len,
	            type_batch, type_len):
		mid_features = []
		attention_mask = []
		if self.ce_flag:
			ce = self.character_encoder(character_batch) # batch * max_character * embedding_size
			mid_features.append(F.pad(ce, [0, self.max_input_dim - self.ce_dim]))
			attention_mask.append(F.pad(torch.ones_like(ce), [0, self.max_input_dim - self.ce_dim]))
		if self.we_flag:
			we = self.word_encoder(word_batch)
			mid_features.append(F.pad(we, [0, self.max_input_dim - self.we_dim]))
			attention_mask.append(F.pad(torch.ones_like(we), [0, self.max_input_dim - self.we_dim]))
		if self.wce_flag:
			wce = self.word_context_encoder(left_word_context_batch, left_word_context_len, right_word_context_batch, right_word_context_len)
			mid_features.append(F.pad(wce, [0, self.max_input_dim - self.wce_dim]))
			attention_mask.append(F.pad(torch.ones_like(wce), [0, self.max_input_dim - self.wce_dim]))
		if self.ee_flag:
			ece = self.entity_context_encoder(left_entity_context_batch, left_entity_context_len, right_entity_context_batch, right_entity_context_len)
			mid_features.append(F.pad(ece, [0, self.max_input_dim - self.ee_dim]))
			attention_mask.append(F.pad(torch.ones_like(ece), [0, self.max_input_dim - self.ee_dim]))
		if self.re_flag:
			re = self.relation_encoder(relation_batch)
			mid_features.append(F.pad(re, [0, self.max_input_dim - self.re_dim]))
			attention_mask.append(F.pad(torch.ones_like(re), [0, self.max_input_dim - self.re_dim]))
		if self.te_flag:
			te = self.type_encoder(type_batch)
			# print(self.type_encoder.out_size, te.size())
			mid_features.append(F.pad(te, [0, self.max_input_dim - self.te_dim]))
			attention_mask.append(F.pad(torch.ones_like(te), [0, self.max_input_dim - self.te_dim]))
		# for item in mid_features: print(item.size())
		ffnn_input = torch.cat(mid_features, dim=-1)
		attention_mask = torch.cat(attention_mask, dim=-1)
		# ffnn_output = self.transformer(ffnn_input)
		binary_output = self.binary_encoder(ffnn_input)
		# print(ffnn_input.size(), binary_output.size())
		return binary_output, ffnn_input, attention_mask

# class JointEntityEncoder(nn.Module):
# 	def __init__(self, use_character_embedding, use_word_context_embedding, use_entity_context_embedding, use_relation_embedding, use_type_embedding,
# 	             character_encoder, word_encoder, entity_encoder, relation_encoder, type_encoder,
# 	             character_embedding_dim, word_embedding_dim, entity_embedding_dim, relation_embedding_dim, type_embedding_dim,
# 	             character_encoding_dim, word_encoding_dim, entity_encoding_dim, relation_encoding_dim, type_encoding_dim):
# 		super(JointEntityEncoder, self).__init__()
#
# 	def forward(self, character_batch, character_len,
# 	            left_word_context_batch, left_word_context_len,
# 	            right_word_context_batch, right_word_context_len,
# 	            left_entity_context_batch, left_entity_context_len,
# 	            right_entity_context_batch, right_entity_context_len,
# 	            relation_batch, relation_len,
# 	            type_batch, type_len):
# 		pass

class VectorTransformer(nn.Module):
	def __init__(self, in_dim, out_dim, features):
		super(VectorTransformer, self).__init__()
		self.transformer = FFNNEncoder(in_dim * features, out_dim, out_dim, 3)
		# self.dropout = nn.Dropout()
		# self.linear = nn.Linear(in_dim * features, out_dim)
		# self.linear_in_dim = in_dim * features

	def forward(self, vec, attention_mask=None):
		return self.transformer(vec)
		# transformed = self.transformer(vec, attention_mask)[0]
		# transformed = transformed.view(-1, self.linear_in_dim)
		# return self.linear(self.dropout(transformed))
