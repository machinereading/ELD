import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sklearn.metrics as metrics
from ...utils import readfile, TimeUtil
from ... import GlobalValues as gl
from . import ModelFactory
from .modules.Scorer import *
from tqdm import tqdm
import logging

class ThreeScorerModel(nn.Module):
	def __init__(self, args):
		super(ThreeScorerModel, self).__init__()
		we = np.load(args.word_embedding_path+".npy")
		we = np.vstack([np.zeros([2, we.shape[1]]), we])
		ee = np.load(args.entity_embedding_path+".npy")
		ee = np.vstack([np.zeros([2, ee.shape[1]]), ee])
		self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(we))
		self.entity_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(ee))
		we_dim = self.word_embedding.embedding_dim
		ee_dim = self.entity_embedding.embedding_dim
		self.er_path = args.er_model_path
		self.el_path = args.el_model_path
		self.ec_path = args.ec_model_path
		self.pretrain_epoch = args.pretrain_epoch
		self.er_score_threshold = getattr(args, "er_score_threshold", 0.5)
		self.el_score_threshold = getattr(args, "el_score_threshold", 0.5)
		self.pretrain_er = True
		self.pretrain_el = True

		# initialize and load model
		self.er_scorer = ERScorer(args, we_dim)
		try:
			self.er_scorer.load_state_dict(torch.load(self.er_path))
			self.pretrain_er = False
		except:
			logging.info("Failed to load ER scorer from %s" % args.er_model_path)
		
		self.el_scorer = ELScorer(args, ee_dim)
		try:
			self.el_scorer.load_state_dict(torch.load(self.el_path))
			self.pretrain_el = False
		except:
			logging.info("Failed to load EL scorer from %s" % args.el_model_path)
		self.ec_scorer = ECScorer(args)
		try:
			self.ec_scorer.load_state_dict(torch.load(self.ec_path))
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

		wctx = torch.stack([rctx_word_batch, lctx_word_batch], dim=0)
		ectx = torch.stack([rctx_entity_batch, lctx_entity_batch], dim=0)
		# score with scorer
		# filter under threshold score
		er_score = F.relu(self.er_scorer(wctx) - self.er_score_threshold) + self.er_score_threshold
		el_score = F.relu(self.el_scorer(ectx) - self.el_score_threshold) + self.er_score_threshold
		ec_score = self.ec_scorer(er_score, el_score, wctx, ectx)
		new_cluster = ec_score * 0 # TODO
		final_score = self.cluster_scorer(torch.FloatTensor([er_score, el_score, ec_score]))
		# final_score = F.sigmoid(final_score)
		# removed because binary_cross_entropy_with_logits applies sigmoid
		return final_score

	def loss(self, prediction, label):
		return F.binary_cross_entropy_with_logits(prediction, label)

	@TimeUtil.measure_time
	def pretrain(self, dataset):
		# pretrain er scorer, el scorer, ec transformer
		logging.info("Pretraining Entity Scorer")
		if self.pretrain_er or self.pretrain_el:
			best_er_f1 = 0
			best_el_f1 = 0
			er_optimizer = torch.optim.Adam(self.er_scorer.parameters())
			el_optimizer = torch.optim.Adam(self.el_scorer.parameters())
			for epoch in tqdm(range(self.pretrain_epoch), desc="Pretraining"):
				dev_batch = []
				c = 1
				for batch in dataset.get_token_batch():
					if c % 10 == 0: 
						dev_batch.append(batch)
						continue
					if c * len(batch) > 10000: break # TODO 10000 may change: limit pretraining size
					if self.pretrain_er:
						self.er_scorer.train()
						lw = self.word_embedding(torch.stack([x.lctxw_ind for x in batch], 0))
						rw = self.word_embedding(torch.stack([x.rctxw_ind for x in batch], 0))
						print(lw.size(), rw.size())
						er_label = [1 if x.is_entity else 0 for x in batch]
						er_optimizer.zero_grad()
						er_pred = self.er_scorer(lw, rw)
						er_loss = self.er_scorer.loss(er_pred, er_label)
						er_loss.backward()
						er_optimizer.step()
					if self.pretrain_el:
						self.el_scorer.train()
						le = self.word_embedding(torch.stack([x.lctxe_ind for x in batch if x.is_entity], 0))
						re = self.word_embedding(torch.stack([x.rctxe_ind for x in batch if x.is_entity], 0))
						el_label = [1 if x.entity_in_kb else 0 for x in batch if x.is_entity]
						el_optimizer.zero_grad()
						el_pred = self.el_scorer(torch.FloatTensor(le), torch.FloatTensor(re))
						el_loss = self.el_scorer.loss(el_pred, el_label)
						el_loss.backward()
						el_optimizer.step()
					c += 1
				if epoch % 5 == 0:
					for batch in dev_batch:
						if self.pretrain_er:
							self.er_scorer.eval()
							lw = self.word_embedding(torch.stack([x.lctxw_ind for x in batch], 0))
							rw = self.word_embedding(torch.stack([x.rctxw_ind for x in batch], 0))
							er_label = [1 if x.is_entity else 0 for x in batch]
							prediction = [1 if x > 0.5 else 0 for x in self.er_scorer(lw, rw)]
							f1 = metrics.f1_score(er_label, prediction)
							print("Epoch %d: ER F1 %.2f" % (epoch, f1))
							if f1 > best_er_f1:
								best_er_f1 = f1
								torch.save(self.er_scorer.save_state_dict(), self.er_path)
						if self.pretrain_el:
							self.el_scorer.eval()
							le = self.word_embedding(torch.stack([x.lctxe_ind for x in batch if x.is_entity], 0))
							re = self.word_embedding(torch.stack([x.rctxe_ind for x in batch if x.is_entity], 0))
							el_label = [1 if x.entity_in_kb else 0 for x in batch if x.is_entity]
							prediction = [1 if x > 0.5 else 0 for x in self.el_scorer(le, re)]
							f1 = metrics.f1_score(el_label, prediction)
							print("Epoch %d: EL F1 %.2f" % (epoch, f1))
							if f1 > best_el_f1:
								best_el_f1 = f1
								torch.save(self.el_scorer.save_state_dict(), self.el_path)

			
			if self.pretrain_el:
				torch.save(self.el_scorer.save_state_dict(), self.el_path)
		logging.info("Pretraining done")

		# fix parameters
		for param in self.er_scorer:
			param.requires_grad = False
		for param in self.el_scorer:
			param.requires_grad = False
