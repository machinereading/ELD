import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ...utils import readfile, KoreanUtil
from ... import GlobalValues as gl
from .modules import ContextEmbedder, Transformer
from .modules.Embedding import Embedding


import logging

class ValidationModel(nn.Module):
	def __init__(self, args):
		super(ValidationModel, self).__init__()
		self.pretrain_epoch = args.pretrain_epoch
		self.pretrain_transformer = True
		self.kb = None
		self.window_size = args.ctx_window_size
		self.target_device = args.device

		self.we = Embedding(args.word_embedding_type, args.word_embedding_path).to(self.target_device)
		self.ee = Embedding(args.entity_embedding_type, args.entity_embedding_path).to(self.target_device)
		self.ce = nn.Embedding(KoreanUtil.jamo_len + len(KoreanUtil.alpha), args.char_embedding_dim)
		self.we_dim = self.we.embedding_dim
		self.ee_dim = self.ee.embedding_dim
		self.ce_dim = args.char_embedding_dim
		
		self.transformer_type = args.transformer
		self.cluster_transformer = Transformer.get_transformer(args)
		self.encode_sequence = args.encode_sequence
		# sequential encoding
		if self.encode_sequence:
			self.jamo_embedder = ContextEmbedder.CNNEmbedder(args.max_jamo, 2, 2)
			self.cw_embedder = ContextEmbedder.BiContextEmbedder(args.er_model, self.we_dim, args.er_output_dim)
			self.ce_embedder = ContextEmbedder.BiContextEmbedder(args.el_model, self.ee_dim, args.el_output_dim)
		# final prediction layer
		self.predict = nn.Sequential(
			nn.Linear(args.transform_dim, 100),
			nn.ReLU(),
			nn.Linear(100, 1),
			nn.Sigmoid()
		)
		try:
			self.cluster_transformer.load_state_dict(torch.load(args.transformer_model_path))
			self.pretrain_transformer = False
		except:
			logging.info("Failed to load transformer from %s" % args.transformer_model_path)

	def forward(self, jamo_index, cluster_word_lctx, cluster_word_rctx, cluster_entity_lctx, cluster_entity_rctx):
		"""
		Input
			jamo_index: Tensor of batch_size * max_vocab_size * max jamo size
			cluster_word/entity_lctx/rctx: Tensor of batch size * max vocab size * window size
		Output
			Confidence score?
		"""
		# cluster -> representation
		print(jamo_index.size(), cluster_word_lctx.size(), cluster_entity_lctx.size())
		jamo_size = jamo_index.size()[-1]
		c_embedding = self.ce(jamo_index).view(-1, jamo_size, self.ce_dim) # batch_size * max_vocab_size * max_jamo_size * jamo_embedding
		voca_size = jamo_index.size()[1]
		# bert attention mask?
		wlctx = self.we(cluster_word_lctx.view(-1, self.window_size)).view(-1, self.window_size, self.we_dim) # batch_size * max_vocab_size * window * embedding_size
		wrctx = self.we(cluster_word_rctx.view(-1, self.window_size)).view(-1, self.window_size, self.we_dim)
		elctx = self.ee(cluster_entity_lctx).view(-1, self.window_size, self.ee_dim)
		erctx = self.ee(cluster_entity_rctx).view(-1, self.window_size, self.ee_dim)
		print(c_embedding.size(), wlctx.size(), elctx.size())
		
		if self.encode_sequence:
			c_embedding = self.jamo_embedder(c_embedding)
			c_embedding = c_embedding.view(-1, voca_size, c_embedding.size()[-1])
			w_embedding = self.cw_embedder(wlctx, wrctx) # (batch_size * max_vocab_size) * embedding size
			print(w_embedding.size())
			w_embedding = w_embedding.view(-1, voca_size, w_embedding.size()[-1]) # batch_size * max_vocab_size * embedding size - 각각의 token마다 embedding dimension의 context embedding 하나씩 들고있음
			print(w_embedding.size())
			e_embedding = self.ce_embedder(elctx, erctx)
			e_embedding = e_embedding.view(-1, voca_size, e_embedding.size()[-1])
		else:
			w_embedding = torch.cat([wlctx, wrctx], 2)
			e_embedding = torch.cat([elctx, erctx], 2)

		
		cluster_representation = self.cluster_transformer(c_embedding, w_embedding, e_embedding) # batch size * cluster representation size
		
		# prediction - let's try simple FFNN this time
		return self.predict(cluster_representation)

	def loss(self, prediction, label):
		return F.binart_cross_entropy(prediction, label)

	def pretrain(self, dataset):
		# pretrain transformer
		if self.pretrain_transformer:
			for epoch in tqdm(range(self.pretrain_epoch)):
				pass

		for param in self.cluster_transformer:
			param.requires_grad = False