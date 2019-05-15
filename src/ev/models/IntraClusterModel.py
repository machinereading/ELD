import logging

import numpy as np
import sklearn.metrics as metrics
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.Embedding import Embedding
from .modules.Scorer import *
from ...utils import jsondump, TimeUtil

class ThreeScorerModel(nn.Module):
	def __init__(self, args):
		super(ThreeScorerModel, self).__init__()
		self.target_device = args.device
		we = np.load(args.word_embedding_path + ".npy")
		we = np.vstack([np.zeros([2, we.shape[1]]), we])
		ee = np.load(args.entity_embedding_path + ".npy")
		ee = np.vstack([np.zeros([2, ee.shape[1]]), ee])
		self.word_embedding = Embedding.load_embedding(args.word_embedding_type, args.word_embedding_path).to(self.target_device)
		self.entity_embedding = Embedding.load_embedding(args.entity_embedding_type, args.entity_embedding_path).to(self.target_device)
		# self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(we)).to(self.target_device)
		# self.entity_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(ee)).to(self.target_device)
		we_dim = self.word_embedding.embedding_dim
		ee_dim = self.entity_embedding.embedding_dim
		self.er_path = args.er_model_path
		self.el_path = args.el_model_path
		self.ec_path = args.ec_model_path
		self.pretrain_epoch = args.pretrain_epoch
		self.pretrain_batch_size = args.batch_size
		self.er_score_threshold = getattr(args, "er_score_threshold", 0.5)
		self.el_score_threshold = getattr(args, "el_score_threshold", 0.5)
		self.pretrain_er = True
		self.pretrain_el = True

		# initialize and load model
		self.er_scorer = ERScorer(args, we_dim).to(self.target_device)
		try:
			self.er_scorer.load_state_dict(torch.load(self.er_path))
			self.pretrain_er = False
		except:
			logging.info("Failed to load ER scorer from %s" % args.er_model_path)

		self.el_scorer = ELScorer(args, ee_dim).to(self.target_device)
		try:
			self.el_scorer.load_state_dict(torch.load(self.el_path))
			self.pretrain_el = False
		except:
			logging.info("Failed to load EL scorer from %s" % args.el_model_path)
		self.ec_scorer = ECScorer(args, self.er_scorer, self.el_scorer).to(self.target_device)
		try:
			self.ec_scorer.load_state_dict(torch.load(self.ec_path))
		except:
			logging.info("Failed to load EC scorer from %s" % args.ec_model_path)
		self.cluster_scorer = nn.Linear(3, 1).to(self.target_device)

		if args.force_pretrain:
			self.pretrain_er = True
			self.pretrain_el = True

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
		new_cluster = ec_score * 0
		final_score = self.cluster_scorer(torch.FloatTensor([er_score, el_score, ec_score]))
		# final_score = F.sigmoid(final_score)
		# removed because binary_cross_entropy_with_logits applies sigmoid
		return final_score

	def loss(self, prediction, label):
		return F.binary_cross_entropy_with_logits(prediction, label)

	@TimeUtil.measure_time
	def pretrain(self, train_dataset, dev_dataset):
		# pretrain er scorer, el scorer, ec transformer
		logging.info("Pretraining Entity Scorer")
		if self.pretrain_er or self.pretrain_el:
			train_dataloader = DataLoader(train_dataset, batch_size=self.pretrain_batch_size, shuffle=True)
			dev_dataloader = DataLoader(dev_dataset, batch_size=self.pretrain_batch_size, shuffle=False)
			best_er_f1 = 0
			best_el_f1 = 0
			er_optimizer = torch.optim.Adam(self.er_scorer.parameters())
			el_optimizer = torch.optim.Adam(self.el_scorer.parameters())
			for epoch in tqdm(range(1, self.pretrain_epoch + 1), desc="Pretraining"):
				dev_batch = []
				for lctxw_ind, rctxw_ind, lctxe_ind, rctxe_ind, error_type in train_dataloader:
					lctxw_ind, rctxw_ind, lctxe_ind, rctxe_ind, error_type = lctxw_ind.to(
						self.target_device), rctxw_ind.to(self.target_device), lctxe_ind.to(
						self.target_device), rctxe_ind.to(self.target_device), error_type.to(self.target_device)

					if self.pretrain_er:
						self.er_scorer.train()
						# print(lctxw_ind.size())
						lw = self.word_embedding(lctxw_ind)
						rw = self.word_embedding(rctxw_ind)
						# print(lw.size())
						er_label = torch.unsqueeze(torch.Tensor([1 if x != 1 else 0 for x in error_type]), 1).to(
							self.target_device)
						er_optimizer.zero_grad()
						er_pred = self.er_scorer(lw, rw)  # batch * 1 ???
						er_loss = self.er_scorer.loss(er_pred, er_label)
						er_loss.backward()
						er_optimizer.step()
					if self.pretrain_el:
						self.el_scorer.train()
						le = self.entity_embedding(lctxe_ind)
						re = self.entity_embedding(rctxe_ind)
						el_label = torch.unsqueeze(torch.Tensor([0 if x != 0 else 1 for x in error_type]), 1).to(
							self.target_device)
						el_optimizer.zero_grad()
						el_pred = self.el_scorer(le, re)
						el_loss = self.el_scorer.loss(el_pred, el_label)
						el_loss.backward()
						el_optimizer.step()

				if epoch % 5 == 0:
					er_label = []
					er_pred = []
					el_label = []
					el_pred = []
					toks = []
					for lctxw_ind, rctxw_ind, lctxe_ind, rctxe_ind, error_type in dev_dataloader:
						lctxw_ind, rctxw_ind, lctxe_ind, rctxe_ind, error_type = lctxw_ind.to(
							self.target_device), rctxw_ind.to(self.target_device), lctxe_ind.to(
							self.target_device), rctxe_ind.to(self.target_device), error_type.to(self.target_device)
						if self.pretrain_er:
							self.er_scorer.eval()
							lw = self.word_embedding(lctxw_ind)
							rw = self.word_embedding(rctxw_ind)
							er_label += [1 if x != 1 else 0 for x in error_type]
							er_pred += [1 if x > 0.5 else 0 for x in self.er_scorer(lw, rw)]

						if self.pretrain_el:
							self.el_scorer.eval()
							le = self.entity_embedding(lctxe_ind)
							re = self.entity_embedding(rctxe_ind)
							el_label += [0 if x != 0 else 1 for x in error_type]
							el_pred += [1 if x > 0.5 else 0 for x in self.el_scorer(le, re)]

					if self.pretrain_er:
						f1 = metrics.f1_score(er_label, er_pred)
						print("Epoch %d: ER F1 %f" % (epoch, f1))
						if f1 > best_er_f1:
							best_er_f1 = f1
							torch.save(self.er_scorer.state_dict(), self.er_path)
					if self.pretrain_el:
						f1 = metrics.f1_score(el_label, el_pred)
						print("Epoch %d: EL F1 %f" % (epoch, f1))
						if f1 > best_el_f1:
							best_el_f1 = f1
							torch.save(self.el_scorer.state_dict(), self.el_path)
							wrong_inst = []

							for p, l, t in zip(el_pred, el_label, toks):
								if p != l:
									wrong_inst.append(t.to_json())
							jsondump(wrong_inst, "runs/ev/el_wrong_%d.json" % epoch)
		logging.info("Pretraining done")

		# fix parameters
		for param in self.er_scorer.parameters():
			param.requires_grad = False
		for param in self.el_scorer.parameters():
			param.requires_grad = False

class JointScorerModel(nn.Module):
	def __init__(self, args):
		super(JointScorerModel, self).__init__()
		self.target_device = args.device
		self.word_embedding = Embedding.load_embedding(args.word_embedding_type, args.word_embedding_path).to(self.target_device)
		self.entity_embedding = Embedding.load_embedding(args.entity_embedding_type, args.entity_embedding_path).to(self.target_device)
		# print(self.word_embedding)
		we_dim = self.word_embedding.embedding_dim
		ee_dim = self.entity_embedding.embedding_dim
		self.model_path = args.joint_model_path
		self.pretrain_epoch = args.pretrain_epoch
		self.pretrain_batch_size = args.batch_size
		self.er_score_threshold = getattr(args, "er_score_threshold", 0.5)
		self.el_score_threshold = getattr(args, "el_score_threshold", 0.5)
		self.pretrain_model = True
		self.scorer = JointScorer(args, we_dim, ee_dim).to(self.target_device)
		try:
			self.scorer.load_state_dict(torch.load(self.model_path))
			self.pretrain_model = False
		except:
			logging.info("Failed to load Joint scorer from %s" % args.joint_model_path)

		if args.force_pretrain:
			self.pretrain_model = True

	def forward(self, batch):
		pass

	@TimeUtil.measure_time
	def pretrain(self, train_dataset, dev_dataset):
		if self.pretrain_model:
			train_dataloader = DataLoader(train_dataset, batch_size=self.pretrain_batch_size, shuffle=True)
			dev_dataloader = DataLoader(dev_dataset, batch_size=self.pretrain_batch_size, shuffle=False)
			best_micro_f1 = 0
			best_macro_f1 = 0
			best_epoch = 0
			optimizer = torch.optim.Adam(self.scorer.parameters())
			for epoch in tqdm(range(1, self.pretrain_epoch + 1), desc="Pretraining"):
				self.scorer.train()
				for lctxw_ind, rctxw_ind, lctxe_ind, rctxe_ind, error_type in train_dataloader:
					lctxw_ind, rctxw_ind, lctxe_ind, rctxe_ind, error_type = lctxw_ind.to(
						self.target_device), rctxw_ind.to(self.target_device), lctxe_ind.to(
						self.target_device), rctxe_ind.to(self.target_device), error_type.to(self.target_device)

					lw = self.word_embedding(lctxw_ind)
					rw = self.word_embedding(rctxw_ind)
					le = self.entity_embedding(lctxe_ind)
					re = self.entity_embedding(rctxe_ind)

					label = error_type
					pred = self.scorer(lw, rw, le, re)
					loss = self.scorer.loss(pred, label)
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()
				if epoch % 5 == 0:
					label = []
					pred = []
					toks = []
					self.scorer.eval()
					with torch.no_grad():
						for lctxw_ind, rctxw_ind, lctxe_ind, rctxe_ind, error_type in dev_dataloader:
							lctxw_ind, rctxw_ind, lctxe_ind, rctxe_ind, error_type = lctxw_ind.to(
								self.target_device), rctxw_ind.to(self.target_device), lctxe_ind.to(
								self.target_device), rctxe_ind.to(self.target_device), error_type.to(self.target_device)
							lw = self.word_embedding(lctxw_ind)
							rw = self.word_embedding(rctxw_ind)
							le = self.entity_embedding(lctxe_ind)
							re = self.entity_embedding(rctxe_ind)
							# print(lw.size(), rw.size(), le.size(), re.size())
							l_batch = [x.item() for x in error_type]
							p_batch = [ind.item() for ind in self.scorer(lw, rw, le, re).max(1)[1]]
							label += l_batch
							pred += p_batch
						# for l, p in zip(l_batch, p_batch):
						# 	print(l, p)

						micro_f1 = metrics.f1_score(label, pred, average="micro")
						macro_f1 = metrics.f1_score(label, pred, average="macro")
						print("Epoch %d: Micro F1 %f, Macro F1 %f" % (epoch, micro_f1, macro_f1))
						print("Best Epoch: Micro F1 %f, Macro F1 %f @ Epoch %d" % (
						best_micro_f1, best_macro_f1, best_epoch))

						if micro_f1 > best_micro_f1:
							best_micro_f1 = micro_f1
							best_macro_f1 = macro_f1
							best_epoch = epoch
							torch.save(self.scorer.state_dict(), self.model_path)
