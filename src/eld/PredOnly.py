import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils.Dataset import ELDDataset, SkipgramDataset
from .modules import CNNVectorTransformer, SeparateEntityEncoder
from .utils import ELDArgs, DataModule
from ..utils import readfile, jsondump

from .. import GlobalValues as gl

class SkipGramEntEmbedding:
	def __init__(self, mode: str, model_name: str, args: ELDArgs, data: DataModule = None):
		gl.logger.info("Initializing prediction model")
		self.mode = mode
		self.model_name = model_name
		self.args = args
		self.device = "cuda"
		if self.mode == "train":
			self.encoder: nn.Module = SeparateEntityEncoder(self.args)
			self.transformer: nn.Module = CNNVectorTransformer(self.encoder.max_input_dim, args.e_emb_dim, args.flags)

			self.word_post_transformer = nn.Embedding.from_pretrained(self.data.entity_embedding, freeze=False)
			self.entity_post_transformer_o = nn.Embedding(len(self.data.e2i), args.e_emb_dim) # 주변개체 예측용
			self.entity_post_transformer_i = nn.Embedding(len(self.data.e2i), args.e_emb_dim) # 중심개체 예측용

			jsondump(args.to_json(), "models/eld/%s_args.json" % model_name)
		if self.mode == "pred":
			self.args = ELDArgs.from_json("models/eld/%s_args.json" % model_name)
			self.load_model()
		if self.mode == "demo":
			self.device = "cpu"
		self.encoder.to(self.device)
		self.transformer.to(self.device)
		self.word_post_transformer.to(self.device)
		self.entity_post_transformer_o.to(self.device)
		self.args.device = self.device
		if data is None:
			self.data = DataModule(mode, args)
		else:
			self.data = data
		self.train_method = args.pred_train_mode
		gl.logger.info("Finished prediction model initialization")

	def train(self):
		gl.logger.info("Training prediction model")
		batch_size = 512
		optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.transformer.paramters()) + list(self.word_post_transformer.parameters()) + list(self.entity_post_transformer_o.parameters()), lr=1e-4, weight_decay=1e-4)
		tqdmloop = tqdm(range(1, self.args.epochs + 1))
		train_dataset = SkipgramDataset(self.mode, self.data.corpus, self.args, cand_dict=self.data.surface_ent_dict, filter_list=[x for x in readfile(self.args.train_filter)], limit=self.args.train_corpus_limit)
		dev_dataset = ELDDataset(self.mode, self.data.corpus, self.args, cand_dict=self.data.surface_ent_dict, filter_list=[x for x in readfile(self.args.dev_filter)], limit=self.args.train_corpus_limit)
		test_dataset = ELDDataset(self.mode, self.data.corpus, self.args, cand_dict=self.data.surface_ent_dict, filter_list=[x for x in readfile(self.args.test_filter)], limit=self.args.train_corpus_limit)

		train_batch = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
		dev_batch = DataLoader(dataset=dev_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

		max_score = 0
		max_score_epoch = 0
		max_score_threshold = 0

		for epoch in tqdmloop:
			self.encoder.train()
			self.transformer.train()
			# nel = []
			# gei = []
			# preds = []
			for batch in train_batch:
				optimizer.zero_grad()
				ce, cl, we, wl, lwe, lwl, rwe, rwl, lee, lel, ree, rel, re, rl, te, tl, _, _, _ = [x.to(self.device, torch.float32) if x is not None else None for x in batch[:-8]]
				args = (ce, cl, we, wl, lwe, lwl, rwe, rwl, lee, lel, ree, rel, re, rl, te, tl)
				# args, kwargs = self.prepare_input(batch)
				_, encoded_mention = self.encoder(*args)
				transformed = self.transformer(encoded_mention)
				pos_word_sample, neg_word_sample, pos_ent_sample, neg_ent_sample = batch[-4:].to(self.device)
				# loss = sum([F.mse_loss(p, g) if torch.sum(g) != 0 else torch.zeros_like(F.mse_loss(p, g)) for p, g in zip(transformed, ee_label)])
				loss = self.skipgramloss(transformed, pos_word_sample, neg_word_sample, pos_ent_sample, neg_ent_sample)

				loss.backward()
				optimizer.step()
				tqdmloop.set_description("Epoch %d - Loss %.4f" % (epoch, float(loss)))

			if epoch % self.args.eval_per_epoch == 0:
				self.encoder.eval()
				self.transformer.eval()
				preds = []
				labels = []
				for batch in dev_batch:
					args, kwargs = self.prepare_input(batch)
					label = batch[-2]
					_, encoded_mention = self.encoder(*args, **kwargs)
					transformed = self.transformer(encoded_mention)
					preds.append(transformed)
					labels.append(label)
				preds = torch.cat(preds, dim=-1)
				labels = torch.cat(labels, dim=-1)
				print()
				ms = (0, 0, 0)

				mi = 0
				for i in range(1, 10):
					pl = [1 if x > 0.1 * i else 0 for x in preds]
					l = [x for x in labels]
					p, r, f, _ = precision_recall_fscore_support(l, pl, average="binary")
					gl.logger.debug("Threshold %.2f: P %.2f R %.2f F %.2f" % (0.1 * i, p * 100, r * 100, f * 100))
					if f > ms[-1]:
						ms = (p, r, f)
						mi = i * 0.1
				p, r, f = ms
				gl.logger.info("Epoch %d max score @ threshold %.2f: P %.2f R %.2f F %.2f" % (epoch, mi, p * 100, r * 100, f * 100))
				if f > max_score[-1]:
					max_score = (p, r, f)
					max_score_epoch = epoch
					max_score_threshold = mi
					self.save_model()
				gl.logger.info("Epoch %d max score @ threshold %.2f: P %.2f R %.2f F %.2f" % (max_score_epoch, max_score_threshold, p * 100, r * 100, f * 100))
		np.save(self.model_name +"_entity_emb", self.entity_post_transformer_i.weight.numpy()) # 이게 진짜 entity embedding으로 동작할 수 있는지?
	def prepare_input(self, batch):
		if self.mode in ["train", "test"]:
			ce, cl, we, wl, lwe, lwl, rwe, rwl, lee, lel, ree, rel, re, rl, te, tl, in_cand_dict_flag, cand_emb, avg_degree = [x.to(self.device, torch.float32) if x is not None else None for x in batch[:-4]]
		else:
			ce, cl, we, wl, lwe, lwl, rwe, rwl, lee, lel, ree, rel, re, rl, te, tl, in_cand_dict_flag, cand_emb, avg_degree = [x.to(self.device, torch.float32) if x is not None else None for x in batch]
		args = (ce, cl, we, wl, lwe, lwl, rwe, rwl, lee, lel, ree, rel, re, rl, te, tl)
		kwargs = {"surface_dict_flag": None, "cand_entities": None, "avg_degree": None}
		if self.args.use_surface_info:
			kwargs["surface_dict_flag"] = in_cand_dict_flag
		if self.args.use_kb_relation_info:
			kwargs["avg_degree"] = avg_degree
		return args, kwargs

	def load_model(self):
		self.encoder: nn.Module = SeparateEntityEncoder(self.args)
		self.transformer: nn.Module = CNNVectorTransformer(self.encoder.max_input_dim, self.args.e_emb_dim, self.args.flags)

		self.word_post_transformer = nn.Embedding.from_pretrained(self.data.entity_embedding, freeze=False)
		self.entity_post_transformer_o = nn.Embedding(len(self.data.e2i), self.args.e_emb_dim)
		load_dict = torch.load(self.args.model_path)
		self.encoder.load_state_dict(load_dict["encoder"])
		self.transformer.load_state_dict(load_dict["transformer"])
		self.word_post_transformer.load_state_dict(load_dict["word_post_transformer"])
		self.entity_post_transformer_i.load_state_dict(load_dict["entity_post_transformer_i"])
		self.entity_post_transformer_o.load_state_dict(load_dict["entity_post_transformer_o"])

	def save_model(self):
		save_dict = {
			"encoder": self.encoder.state_dict(),
			"transformer": self.transformer.state_dict(),
			"word_post_transformer": self.word_post_transformer.state_dict(),
			"entity_post_transformer_o": self.entity_post_transformer_o.state_dict(),
			"entity_post_transformer_i": self.entity_post_transformer_i.state_dict()
		}
		torch.save(save_dict, self.args.model_path)

	def skipgramloss(self, pred_emb, token_pos_sample, token_neg_sample, entity_pos_sample, entity_neg_sample):
		# code from https://github.com/theeluwin/pytorch-sgns/blob/master/model.py
		# tps, tns, eps, ens = [x.transpose(0, 1).to(self.device) for x in (token_pos_sample, token_neg_sample, entity_pos_sample, entity_neg_sample)] # batch * sample_size
		tps, tns = [self.word_post_transformer(x) for x in [token_pos_sample, token_neg_sample]] # batch * sample_size * e_emb_dim
		eps, ens = [self.entity_post_transformer_o(x).neg() for x in [entity_pos_sample, entity_neg_sample]] # batch * sample_size * e_emb_dim
		pred_emb = pred_emb.unsqueeze(2)
		tploss = torch.bmm(tps, pred_emb).squeeze().sigmoid().log().mean(1)
		eploss = torch.bmm(eps, pred_emb).squeeze().sigmoid().log().mean(1)
		tnloss = torch.bmm(tns, pred_emb).squeeze().sigmoid().log().view(-1, token_pos_sample.size(1), token_neg_sample.size(1)).sum(2).mean(1)
		enloss = torch.bmm(ens, pred_emb).squeeze().sigmoid().log().view(-1, entity_pos_sample.size(1), entity_neg_sample.size(1)).sum(2).mean(1)
		return -sum([tploss, tnloss, eploss, enloss]).mean()


	def pred(self, batch, *, eld_items=None, dump_name=""):
		self.encoder.eval()
		self.transformer.eval()
		preds = []
		labels = []
		for batch in batch:
			args, kwargs = self.prepare_input(batch)
			label = batch[-4]
			_, encoded_mention = self.encoder(*args, **kwargs)
			emb = self.transformer(encoded_mention)
			preds.append(emb)
			labels.append(label)
		# print(pred.size())
		preds = torch.cat(preds, dim=0)
		labels = torch.cat(labels, dim=-1)
		# for p, l in zip(preds, labels):
		# 	print(p.item(), l.item())
		print()
		ms = (0, 0, 0)

		mi = 0
		for i in range(1, 10):
			pl = [1 if x > 0.1 * i else 0 for x in preds]
			l = [x for x in labels]
			p, r, f, _ = precision_recall_fscore_support(l, pl, average="binary")
			gl.logger.debug("Threshold %.2f: P %.2f R %.2f F %.2f" % (0.1 * i, p * 100, r * 100, f * 100))
			if f > ms[-1]:
				ms = (p, r, f)
				mi = i * 0.1
		if eld_items is not None and dump_name != "":
			if not os.path.isdir("runs/eld/%s" % self.model_name):
				os.mkdir("runs/eld/%s" % self.model_name)
			jsondump(generate_result_dict(eld_items, preds, labels), "runs/eld/%s/%s_%s.json" % (self.model_name, self.model_name, dump_name))

		return ms, mi

class PEMEntEmbedding:
	pass


def generate_result_dict(eld_items, preds, labels):
	return {}
