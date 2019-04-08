import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ...utils import readfile
from ... import GlobalValues as gl
from . import ModelFactory
from .modules import Scorer


import logging

class ValidationModel(nn.Module):
	def __init__(self, args):
		super(ValidationModel, self).__init__()
		self.cluster_transformer = None
		self.pretrain_epoch = args.pretrain_epoch
		# self.cw_embedder = Scorer.BiContextEREmbedder()
		# self.ce_embedder = Scorer.BiContextELEmbedder()
		self.pretrain_transformer = True
		self.kb = None
		try:
			self.cluster_transformer.load_state_dict(torch.load(args.transformer_model_path))
			self.pretrain_transformer = False
		except:
			logging.info("Failed to load transformer from %s" % args.transformer_model_path)

	def forward(self, cluster_words, cluster_word_lctx, cluster_word_rctx, cluster_entity_lctx, cluster_entity_rctx):
		# cluster -> representation
		w_embedding = self.cw_embedder(cluster_word_lctx, cluster_word_rctx)
		e_embedding = self.ce_embedder(cluster_entity_lctx, cluster_entity_rctx)
		
		cluster_representation = self.cluster_transformer(batch)
		

	def loss(self, prediction, label):
		return F.cross_entropy_with_logits(prediction, label)

	def pretrain(self, dataset):
		# pretrain transformer
		if self.pretrain_transformer:
			for epoch in tqdm(range(self.pretrain_epoch)):
				pass

		for param in self.cluster_transformer:
			param.requires_grad = False