import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.ds import Corpus
from src.eld.utils.Dataset import ELDDataset
from ..utils import jsondump
from .modules import SeparateEntityEncoder
from .utils import ELDArgs, DataModule
from .. import GlobalValues as gl

class DiscoveryModel:
	def __init__(self, mode: str, model_name: str, args: ELDArgs=None, data: DataModule = None):
		gl.logger.info("Initializing discovery model")
		self.mode = mode
		self.model_name = model_name
		self.args = args
		self.device = "cuda" if self.mode != "demo" else "cpu"

		if self.mode == "train":
			if data is None:
				self.data = DataModule(mode, self.args)
			else:
				self.data = data
			self.model: nn.Module = SeparateEntityEncoder(self.args)
			self.save_model()
			self.model.to(self.device)
			jsondump(self.args.to_json(), "models/eld/%s_args.json" % model_name)
		else:
			self.args = ELDArgs.from_json("models/eld/%s_args.json" % model_name)
			self.load_model()
		self.args.device = self.device
		if self.mode != "train":
			if data is None:
				self.data = DataModule(mode, self.args)
			else:
				self.data = data
		gl.logger.info("Finished discovery model initialization")

	def train(self, train_data, dev_data, test_data=None):
		gl.logger.info("Training discovery model")
		optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
		tqdmloop = tqdm(range(1, self.args.epochs + 1))
		self.data.initialize_corpus_tensor(train_data)
		train_dataset = ELDDataset(self.mode, train_data, self.args, cand_dict=self.data.surface_ent_dict, limit=self.args.train_corpus_limit)
		train_batch = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True, num_workers=4)

		self.data.initialize_corpus_tensor(dev_data)
		dev_dataset = ELDDataset(self.mode, dev_data, self.args, cand_dict=self.data.surface_ent_dict, limit=self.args.train_corpus_limit)
		dev_batch = DataLoader(dataset=dev_dataset, batch_size=512, shuffle=False, num_workers=4)

		max_score = (0, 0, 0)
		max_score_epoch = 0
		max_score_threshold = 0
		for epoch in tqdmloop:
			self.model.train()

			for batch in train_batch:
				optimizer.zero_grad()
				args, kwargs = self.prepare_input(batch)
				label = batch[-4]
				pred, _ = self.model(*args, **kwargs)
				loss = F.binary_cross_entropy_with_logits(pred.view(-1), label.to(dtype=torch.float, device=self.device), pos_weight=torch.tensor([3]).to(dtype=torch.float, device=self.device))
				loss.backward()
				optimizer.step()
				tqdmloop.set_description("Epoch %d - Loss %.4f" % (epoch, float(loss)))
			if epoch % self.args.eval_per_epoch == 0:
				(p, r, f), mi = self.pred(dev_batch, eld_items=dev_dataset.eld_items, dump_name=str(epoch))
				gl.logger.info("Epoch %d max score @ threshold %.2f: P %.2f R %.2f F %.2f" % (epoch, mi, p * 100, r * 100, f * 100))
				if f > max_score[-1]:
					max_score = (p, r, f)
					max_score_epoch = epoch
					max_score_threshold = mi
					self.save_model()
				gl.logger.info("Epoch %d max score @ threshold %.2f: P %.2f R %.2f F %.2f" % (max_score_epoch, max_score_threshold, p * 100, r * 100, f * 100))
				if not os.path.isdir("runs/eld/%s" % self.model_name):
					os.mkdir("runs/eld/%s" % self.model_name)
				if self.args.early_stop <= epoch - max_score_epoch:
					break
		if test_data is not None:
			jsondump(self.test(test_data), "runs/eld/%s/%s_test.json" % (self.model_name, self.model_name))

	def test(self, test_data):
		self.load_model()
		with torch.no_grad():
			self.model.eval()
			self.data.initialize_corpus_tensor(test_data)
			test_dataset = ELDDataset(self.mode, test_data, self.args, cand_dict=self.data.surface_ent_dict, limit=self.args.train_corpus_limit)
			test_batch = DataLoader(dataset=test_dataset, batch_size=512, shuffle=False, num_workers=4)
			result_dict = self.pred(test_batch, eld_items=test_dataset.eld_items, return_as_dict=True)
			return result_dict
			# p, r, f = test_score
			# gl.logger.info("Test score @ threshold %.2f: P %.2f R %.2f F %.2f" % (test_threshold, p * 100, r * 100, f * 100))
			# self.args.out_kb_threshold = test_threshold
			# jsondump(self.args.to_json(), self.args.arg_path)

	def load_model(self):
		self.model: nn.Module = SeparateEntityEncoder(self.args)
		if self.mode != "demo":
			self.model.load_state_dict(torch.load(self.args.model_path))
		else:
			self.model.load_state_dict(torch.load(self.args.model_path, map_location=lambda storage, location: storage))
		self.model.to(self.device)

	def save_model(self):
		jsondump(self.args.to_json(), self.args.arg_path)
		torch.save(self.model.state_dict(), self.args.model_path)

	def prepare_input(self, batch):
		if self.mode  == "train":
			ce, cl, we, wl, lwe, lwl, rwe, rwl, lee, lel, ree, rel, re, rl, te, tl, in_cand_dict_flag, avg_degree = [x.to(self.device, torch.float32) if x is not None else None for x in batch[:-6]]
		elif self.mode  == "test":
			ce, cl, we, wl, lwe, lwl, rwe, rwl, lee, lel, ree, rel, re, rl, te, tl, in_cand_dict_flag, avg_degree = [x.to(self.device, torch.float32) if x is not None else None for x in batch[:-4]]
		else:
			ce, cl, we, wl, lwe, lwl, rwe, rwl, lee, lel, ree, rel, re, rl, te, tl, in_cand_dict_flag, avg_degree = [x.to(self.device, torch.float32) if x is not None else None for x in batch]
		args = (ce, cl, we, wl, lwe, lwl, rwe, rwl, lee, lel, ree, rel, re, rl, te, tl)
		kwargs = {"surface_dict_flag": None, "cand_entities": None, "avg_degree": None}
		if self.args.use_surface_info:
			kwargs["surface_dict_flag"] = in_cand_dict_flag
		if self.args.use_kb_relation_info:
			kwargs["avg_degree"] = avg_degree
		return args, kwargs

	def pred(self, batch, *, pred_mode=False, eld_items=None, dump_name="", return_as_dict=False):
		self.model.eval()
		preds = []
		labels = []
		for batch in batch:
			args, kwargs = self.prepare_input(batch)
			label = batch[-4]
			pred, _ = self.model(*args, **kwargs)
			preds.append(pred.view(-1))
			labels.append(label)
			# print(pred.size())
		preds = torch.sigmoid(torch.cat(preds, dim=-1))
		labels = torch.cat(labels, dim=-1)
		# for p, l in zip(preds, labels):
		# 	print(p.item(), l.item())
		if pred_mode: return preds
		ms = (0, 0, 0)

		mi = 0
		scores = {}
		for i in range(1, 20):
			pl = [1 if x > self.data.calc_threshold(i) else 0 for x in preds]
			l = [x for x in labels]
			p, r, f, _ = precision_recall_fscore_support(l, pl, average="binary")

			if f > ms[-1]:
				ms = (p, r, f)
				mi = i
			scores[self.data.calc_threshold(i)] = [p, r, f]
		gl.logger.debug("Threshold %.2f: P %.2f R %.2f F %.2f" % (self.data.calc_threshold(mi), ms[0] * 100, ms[1] * 100, ms[2] * 100))
		j = generate_result_dict(eld_items, scores, preds, labels)
		if eld_items is not None and dump_name != "":
			if not os.path.isdir("runs/eld/%s" % self.model_name):
				os.mkdir("runs/eld/%s" % self.model_name)
			jsondump(j, "runs/eld/%s/%s_%s.json" % (self.model_name, self.model_name, dump_name))
		if return_as_dict: return j
		return ms, mi


	def __call__(self, data: Corpus, batch_size=512):
		self.data.initialize_corpus_tensor(data)
		dataset = self.data.prepare("pred", data)
		dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=8)
		kb_scores = []

		with torch.no_grad():
			kb_scores.append(self.pred(dataloader, pred_mode=True))
		kb_scores = torch.cat(kb_scores).view(-1)
		for item, kb_score in zip(data.eld_items, kb_scores):
			item.kb_score = kb_score.item()
			item.is_dark_entity = kb_score.item() > self.args.new_ent_threshold
		return data

def generate_result_dict(corpus, score, preds, labels):
	result = {
		"score": score,
		"data":[]
	}
	for e, p, l in zip(corpus, preds, labels):
		if type(p) is torch.Tensor:
			p = float(p.item())
		if type(l) is torch.Tensor:
			l = l.item()
		result["data"].append({
			"Surface"           : e.surface,
			"Context"           : " ".join([x.surface for x in e.lctx[-5:]] + ["[%s]" % e.surface] + [x.surface for x in e.rctx[:5]]),
			"NewEntPred"        : p,
			"NewEntLabel"       : l
		})
	return result
