import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import ELDArgs
from . import BiContextEncoder, CNNEncoder, FFNNEncoder, SelfAttentionEncoder

# entity context emb + relation emb --> transE emb
class SeparateEntityEncoder(nn.Module):
	def __init__(self, args:ELDArgs):
		super(SeparateEntityEncoder, self).__init__()
		# legal = encoder_map.keys()
		# for encoder in [character_encoder, word_encoder, entity_encoder, relation_encoder, type_encoder]:
		# 	assert encoder in legal, "Illegal encoder type: %s, must be one of %s" % (encoder, "/".join(legal))
		self.ce_flag = args.use_character_embedding
		self.we_flag = args.use_word_embedding
		self.wce_flag = args.use_word_context_embedding
		self.ee_flag = args.use_entity_context_embedding
		self.re_flag = args.use_relation_embedding
		self.te_flag = args.use_type_embedding
		assert self.ce_flag or self.we_flag or self.wce_flag or self.ee_flag or self.re_flag or self.te_flag  # use at least one flag
		self.transformer_layer = 2

		self.transformer_hidden_dim = 300
		self.transformer_output_dim = args.e_emb_dim

		self.transformer_input_dim = 0
		separate_layers = len([x for x in [self.ce_flag, self.we_flag, self.wce_flag, self.ee_flag, self.re_flag, self.te_flag] if x])
		self.ce_dim = self.we_dim = self.wce_dim = self.ee_dim = self.re_dim = self.te_dim = 0
		# print(character_embedding_dim, word_embedding_dim, entity_embedding_dim, relation_embedding_dim, type_embedding_dim)
		if self.ce_flag:
			if args.character_encoder.lower() == "cnn":
				self.character_encoder = CNNEncoder(args.c_emb_dim, args.jamo_limit, 2, 3)
				character_encoding_dim = self.character_encoder.out_size
				self.ce_dim = character_encoding_dim
			elif args.character_encoder.lower() == "selfattn":
				self.character_encoder = SelfAttentionEncoder(args.c_emb_dim, 6, 5, args.jamo_limit, output_dim=args.c_enc_dim)
				self.ce_dim = args.c_enc_dim
		if self.we_flag:
			if args.word_encoder.lower() == "cnn":
				self.word_encoder = CNNEncoder(args.w_emb_dim, args.word_limit, 2, 3)
				word_encoding_dim = self.word_encoder.out_size
				self.we_dim = word_encoding_dim
			elif args.word_encoder.lower() == "selfattn":
				self.word_encoder = SelfAttentionEncoder(args.w_emb_dim, 6, 5, args.word_limit, output_dim=args.w_enc_dim)
				self.we_dim = args.w_enc_dim
		if self.wce_flag:
			if args.word_context_encoder.lower() == "bilstm":
				self.wce_dim = args.wc_enc_dim * 2
				self.word_context_encoder = BiContextEncoder("LSTM", args.w_emb_dim, args.wc_enc_dim)
		if self.ee_flag:
			if args.entity_context_encoder.lower() == "bilstm":
				self.ee_dim = args.ec_enc_dim * 2
				self.entity_context_encoder = BiContextEncoder("LSTM", args.e_emb_dim, args.ec_enc_dim)
		if self.re_flag:
			if args.relation_encoder.lower() == "cnn":
				self.relation_encoder = CNNEncoder(args.r_emb_dim, args.relation_limit, 2, 2)
				relation_encoding_dim = self.relation_encoder.out_size
				self.re_dim = relation_encoding_dim
			# elif relation_encoder.lower() == "selfattn":
			# 	self.relation_encoder = SelfAttentionEncoder(relation_embedding_dim, 4, 5, 1, output_dim=relation_encoding_dim)
			# 	self.re_dim = relation_encoding_dim
		if self.te_flag:
			if args.type_encoder.lower() == "ffnn":
				self.type_encoder = FFNNEncoder(args.t_emb_dim, args.t_enc_dim, (args.t_emb_dim + args.t_enc_dim) // 2, 2)
				self.te_dim = args.t_enc_dim
			elif args.type_encoder.lower() == "selfattn":
				self.type_encoder = SelfAttentionEncoder(args.t_emb_dim, 6, 5, 1, output_dim=args.t_enc_dim)
				self.te_dim = args.t_enc_dim

		self.max_input_dim = max(self.ce_dim, self.we_dim, self.wce_dim, self.ee_dim, self.re_dim, self.te_dim)
		self.transformer_input_dim = self.max_input_dim * separate_layers

		# seq = [nn.Linear(self.transformer_input_dim, self.transformer_output_dim), nn.Dropout()] if self.transformer_layer < 2 else \
		# 	[nn.Linear(self.transformer_input_dim, self.transformer_hidden_dim), nn.Dropout()] + [nn.Linear(self.transformer_input_dim, self.transformer_hidden_dim), nn.Dropout()] * (self.transformer_layer - 2) + [
		# 		nn.Linear(self.transformer_hidden_dim, self.transformer_output_dim),
		# 		nn.Dropout()]
		# self.transformer = nn.Sequential(*seq)

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
	            type_batch, type_len, *,
	            surface_dict_flag=None, cand_entities=None):
		mid_features = []
		if self.ce_flag:
			ce = self.character_encoder(character_batch) # batch * max_character * embedding_size
			mid_features.append(F.pad(ce, [0, self.max_input_dim - self.ce_dim]))
		if self.we_flag:
			we = self.word_encoder(word_batch)
			mid_features.append(F.pad(we, [0, self.max_input_dim - self.we_dim]))
		if self.wce_flag:
			wce = self.word_context_encoder(left_word_context_batch, left_word_context_len, right_word_context_batch, right_word_context_len)
			mid_features.append(F.pad(wce, [0, self.max_input_dim - self.wce_dim]))
		if self.ee_flag:
			ece = self.entity_context_encoder(left_entity_context_batch, left_entity_context_len, right_entity_context_batch, right_entity_context_len)
			mid_features.append(F.pad(ece, [0, self.max_input_dim - self.ee_dim]))
		if self.re_flag:
			re = self.relation_encoder(relation_batch)
			mid_features.append(F.pad(re, [0, self.max_input_dim - self.re_dim]))
		if self.te_flag:
			te = self.type_encoder(type_batch)
			# print(self.type_encoder.out_size, te.size())
			mid_features.append(F.pad(te, [0, self.max_input_dim - self.te_dim]))
		if surface_dict_flag is not None:
			mid_features.append(surface_dict_flag)
		if cand_entities is None:
			# for item in mid_features: print(item.size())
			ffnn_input = torch.cat(mid_features, dim=-1)
			# ffnn_output = self.transformer(ffnn_input)
			binary_output = self.binary_encoder(ffnn_input)
			# print(ffnn_input.size(), binary_output.size())
			return binary_output, ffnn_input
		else:
			pass

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

class FFNNVectorTransformer(nn.Module):
	def __init__(self, in_dim, out_dim, features):
		super(FFNNVectorTransformer, self).__init__()
		self.transformer = FFNNEncoder(in_dim * features, out_dim, out_dim, 3)


	def forward(self, vec, attention_mask=None, eval=False):
		return self.transformer(vec)

class SelfAttnVectorTransformer(nn.Module):
	def __init__(self, in_dim, out_dim, features):
		super(SelfAttnVectorTransformer, self).__init__()
		self.transformer = SelfAttentionEncoder(in_dim, 5, 4, features)
		self.dropout = nn.Dropout(p=0.2)
		self.linear = nn.Linear(in_dim * features, out_dim)
		self.linear_in_dim = in_dim * features

	def forward(self, vec, attention_mask=None, eval=False):
		if eval: print("vec", vec)
		transformed = self.transformer(vec, attention_mask, eval=eval)[0]
		if eval: print("transformed", transformed)
		transformed = transformed.view(-1, self.linear_in_dim)
		if eval: print("transformed2", transformed)
		result = self.linear(self.dropout(transformed))
		if eval: print("final", result)
		return result

class CNNVectorTransformer(nn.Module):
	def __init__(self, in_dim, out_dim, features):
		super(CNNVectorTransformer, self).__init__()
		self.features = features
		self.in_dim = in_dim
		self.transformer = CNNEncoder(in_dim, features, features, 3)
		out_size = self.transformer.out_size
		self.dropout = nn.Dropout(p=0.2)
		self.linear = nn.Linear(out_size, out_dim)

	def forward(self, vec, eval=False):
		assert vec.size(-1) == self.features * self.in_dim
		transformed = vec.view(-1, self.features, self.in_dim)
		if eval: print("CNN-before", transformed)
		transformed = self.transformer(transformed)
		if eval: print("CNN-after", transformed)
		result = self.linear(self.dropout(transformed))
		if eval: print("CNN-result", result)
		return result
