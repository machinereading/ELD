import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils import readfile
from ... import GlobalValues as gl
from . import ModelFactory

from .modules.Scorer import ERScorer, ELScorer, ECScorer

class ThreeScoreModel(nn.Module):
	def __init__(self, args):
		super(Model, self).__init__()
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

		
		self.cluster_scorer = nn.Linear(3, 1)
		

	def forward(self, batch):
		# change word index to word embedding
		lctx_word_batch = None
		# score with scorer
		# filter under threshold score
		er_score = F.relu(self.er_scorer() - self.er_score_threshold) + self.er_score_threshold
		el_score = F.relu(self.el_scorer() - self.el_score_threshold) + self.er_score_threshold

		final_score = self.cluster_scorer(torch.FloatTensor([er_score, el_score, ec_score]))
		final_score = F.sigmoid(final_score)

		return final_score
