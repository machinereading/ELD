import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils import readfile, TimeUtil
from ... import GlobalValues as gl
from . import ModelFactory
from .modules.Scorer import ERScorer, ELScorer, ECScorer
from tqdm import tqdm
import logging

class ThreeScoreModel(nn.Module):
	def __init__(self, args):
		super(ThreeScoreModel, self).__init__()
		self.word_embedding = nn.Embedding.from_pretrained(args.word_embedding_path)
		self.entity_embedding = nn.Embedding.from_pretrained(args.entity_embedding_path)
		we_dim = self.word_embedding.shape[1]
		ee_dim = self.entity_embedding.shape[1]
		self.er_score_threshold = getattr(args, er_score_threshold, 0.5)
		self.el_score_threshold = getattr(args, el_score_threshold, 0.5)
		self.pretrain_er = True
		self.pretrain_el = True

		# initialize and load model
		self.er_scorer = ERScorer(args)
		try:
			self.er_scorer.load_state_dict(torch.load(args.er_model_path))
			self.pretrain_er = False
		except:
			logging.info("Failed to load ER scorer from %s" % args.er_model_path)
		
		self.el_scorer = ELScorer(args)
		try:
			self.el_scorer.load_state_dict(torch.load(args.el_model_path))
			self.pretrain_el = False
		except:
			logging.info("Failed to load EL scorer from %s" % args.el_model_path)
		self.ec_scorer = ECScorer(args)
		try:
			self.ec_scorer.load_state_dict(torch.load(args.ec_model_path))
		except:
			logging.info("Failed to load EC scorer from %s" % args.ec_model_path)
		self.cluster_scorer = nn.Linear(3, 1)
		

	def forward(self, batch):
		# change word index to word embedding
		# batch dimension: [batch size(cluster), max cluster size of batch, 2 * window size]
		lctx_word_batch = self.word_embedding(batch["lctx_words"])
		rctx_word_batch = self.word_embedding(batch["rctx_words"])
		lctx_entity_batch = self.entity_embedding(batch["lctx_entities"])
		rctx_entity_batch = self.entity_embedding(batch["rctx_entities"])

		wctx = torch.cat([rctx_word_batch, lctx_word_batch], dim=0)
		ectx = torch.cat([rctx_entity_batch, lctx_entity_batch], dim=0)
		# score with scorer
		# filter under threshold score
		er_score = F.relu(self.er_scorer(wctx) - self.er_score_threshold) + self.er_score_threshold
		el_score = F.relu(self.el_scorer(ectx) - self.el_score_threshold) + self.er_score_threshold
		ec_score = self.ec_scorer(er_score, el_score, wctx, ectx)
		new_cluster = ec_score * 0 # TODO
		final_score = self.cluster_scorer(torch.FloatTensor([er_score, el_score, ec_score]))
		final_score = F.sigmoid(final_score)

		return final_score

	def loss(self, prediction, label):
		pass

	@TimeUtil.measure_time
	def pretrain(self, dataset):
		# pretrain er scorer, el scorer, ec transformer
		logging.info("Pretraining Entity Scorer")
		if self.pretrain_er and self.pretrain_el:
			er_optimizer = torch.optim.Adam(self.er_scorer.parameters())
			el_optimizer = torch.optim.Adam(self.el_scorer.parameters())
			for epoch in tqdm(range(20), desc="Pretraining ER & EL"):
				for batch in dataset.get_token_batch():
					er_optimizer.zero_grad()
					el_optimizer.zero_grad()

					loss = self.
		elif self.pretrain_er:
			er_optimizer = torch.optim.Adam(self.er_scorer.parameters())
			for epoch in tqdm(range(20), desc="Pretraining ER"):
				for batch in dataset.get_token_batch():

		elif self.pretrain_el:
			el_optimizer = torch.optim.Adam(self.el_scorer.parameters())
			


		# fix parameters
		for param in self.er_scorer:
			param.requires_grad = False
		for param in self.el_scorer:
			param.requires_grad = False
