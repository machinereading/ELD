import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils import jsondump
from .modules import SeparateEntityEncoder
from .utils import ELDArgs, DataModule
from .. import GlobalValues as gl

class DiscoveryModel:
	def __init__(self, mode: str, model_name: str, args: ELDArgs, data: DataModule = None):
		gl.logger.info("Initializing discovery model")
		self.mode = mode
		self.model_name = model_name
		self.args = args
		self.device = "cuda"

		if self.mode == "train":
			self.model: nn.Module = SeparateEntityEncoder(self.args)
			self.save_model()
		if self.mode == "pred":
			self.args = ELDArgs.from_json("models/eld/%s_args.json" % model_name)
			self.load_model()
		if self.mode == "demo":
			self.device = "cpu"
		self.model.to(self.device)
		self.args.device = self.device
		if data is None:
			self.data = DataModule(mode, args)
		else:
			self.data = data
		gl.logger.info("Finished discovery model initialization")

	def train(self):
		gl.logger.info("Training discovery model")
		optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
		tqdmloop = tqdm(range(1, self.args.epochs + 1))
		train_batch = DataLoader(dataset=self.data.train_dataset, batch_size=512, shuffle=True, num_workers=4)
		dev_batch = DataLoader(dataset=self.data.dev_dataset, batch_size=512, shuffle=False, num_workers=4)
		test_batch = DataLoader(dataset=self.data.test_dataset, batch_size=512, shuffle=False, num_workers=4)
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
				loss = F.binary_cross_entropy_with_logits(pred.view(-1), label.to(dtype=torch.float, device=self.device))
				loss.backward()
				optimizer.step()
				tqdmloop.set_description("Epoch %d - Loss %.4f" % (epoch, float(loss)))
			if epoch % self.args.eval_per_epoch == 0:
				(p, r, f), mi = self.pred(dev_batch, eld_items=self.data.dev_dataset.eld_items, dump_name=str(epoch))
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
		self.load_model()
		test_score, test_threshold = self.pred(test_batch, eld_items=self.data.test_dataset.eld_items, dump_name="test")
		p, r, f = test_score
		gl.logger.info("Test score @ threshold %.2f: P %.2f R %.2f F %.2f" % (test_threshold, p * 100, r * 100, f * 100))
	def load_model(self):
		self.model: nn.Module = SeparateEntityEncoder(self.args)
		self.model.load_state_dict(torch.load(self.args.model_path))

	def save_model(self):
		torch.save(self.model.state_dict(), self.args.model_path)

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

	def pred(self, batch, *, eld_items=None, dump_name=""):
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
		preds = torch.softmax(torch.cat(preds, dim=-1), dim=-1)
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
		if eld_items is not None and dump_name != "":
			if not os.path.isdir("runs/eld/%s" % self.model_name):
				os.mkdir("runs/eld/%s" % self.model_name)
			jsondump(generate_result_dict(eld_items, ms, preds, labels), "runs/eld/%s/%s_%s.json" % (self.model_name, self.model_name, dump_name))

		return ms, mi

def generate_result_dict(corpus, score, preds, labels):
	result = {
		"score": list(score),
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
