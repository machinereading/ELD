import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils import readfile
from ... import GlobalValues as gl
from . import ModelFactory

from .modules.Scorer import ERScorer, ELScorer, ECScorer

class ThreeScoreModel(nn.Module):
	def __init__(self, args):
		super(ThreeScoreModel, self).__init__()
		self.word_embedding = nn.Embedding.from_pretrained(args.word_embedding_path)
		self.entity_embedding = nn.Embedding.from_pretrained(args.entity_embedding_path)
		we_dim = self.word_embedding.shape[1]
		ee_dim = self.entity_embedding.shape[1]
		self.er_score_threshold = getattr(args, er_score_threshold, 0.5)
		self.el_score_threshold = getattr(args, el_score_threshold, 0.5)


		if args.er_model_path is not None:
			pass
		else:
			self.er_scorer = ERScorer(args)
		if args.el_model_path is not None:
			pass
		else:
			self.el_scorer = ELScorer(args)
		if args.ec_model_path is not None:
			pass
		else:
			self.ec_scorer = ECScorer(args)

		for param in self.er_scorer:
			param.requires_grad = False
		for param in self.el_scorer:
			param.requires_grad = False


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

	def pretrain(self):
		# pretrain er scorer, el scorer, ec transformer
		pass