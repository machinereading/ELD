import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils import readfile
from ... import GlobalValues as gl
from . import ModelFactory

import logging

class ValidationModel(nn.Module):
	def __init__(self, args):
		super(ValidationModel, self).__init__()
		self.cluster_transformer = None
		try:
			self.cluster_transformer.load_state_dict(torch.load(args.transformer_model_path))
			self.pretrain_transformer = False
		except:
			logging.info("Failed to load transformer from %s" % args.transformer_model_path)

	def forward(self, kb, batch):
		pass

	def loss(self, prediction, label):
		return F.cross_entropy(prediction, label)

	def pretrain(self, dataset):
		# pretrain transformer
		if self.pretrain_transformer:
			pass

		for param in self.cluster_transformer:
			param.requires_grad = False