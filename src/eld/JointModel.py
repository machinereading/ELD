import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.ds import Corpus
from src.eld.utils import Evaluator
from src.eld.utils.Dataset import ELDDataset
from ..utils import jsondump
from .modules import SeparateEntityEncoder, CNNVectorTransformer, FFNNVectorTransformer
from .utils import ELDArgs, DataModule
from .. import GlobalValues as gl
class JointModel:
	def __init__(self, mode: str, model_name: str, args: ELDArgs=None, data: DataModule = None):
		gl.logger.info("Initializing joint model")
		self.mode = mode
		self.model_name = model_name
		self.args = args
		self.device = "cuda" if self.mode != "demo" else "cpu"

		if self.mode == "train":
			if data is None:
				self.data = DataModule(mode, self.args)
			else:
				self.data = data
			self.discovery_model: nn.Module = SeparateEntityEncoder(self.args)
			if args.vector_transformer == "cnn":
				self.transformer: nn.Module = CNNVectorTransformer(self.discovery_model.max_input_dim, args.e_emb_dim, args.flags)
			elif args.vector_transformer == "ffnn":
				self.transformer: nn.Module = FFNNVectorTransformer(self.discovery_model.max_input_dim, args.e_emb_dim, args.flags)
			# self.transformer: nn.Module = CNNVectorTransformer(self.discovery_model.max_input_dim, args.e_emb_dim, args.flags)
			self.save_model()
			self.discovery_model.to(self.device)
			self.transformer.to(self.device)
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
		optimizer = torch.optim.Adam(self.discovery_model.parameters(), lr=1e-4, weight_decay=1e-4)
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
		evaluator = Evaluator(self.args, self.data)
		for epoch in tqdmloop:
			self.discovery_model.train()
			self.transformer.train()
			for batch in train_batch:
				optimizer.zero_grad()
				args, kwargs = self.prepare_input(batch)
				label = batch[-4]
				pred, encoded_mention = self.discovery_model(*args, **kwargs)
				transformed = self.transformer(encoded_mention)
				candidate_embs = batch[-6]
				answers = batch[-5]
				ee_label = batch[-3]
				loss = F.binary_cross_entropy_with_logits(pred.view(-1), label.to(dtype=torch.float, device=self.device), pos_weight=torch.tensor([3]).to(dtype=torch.float, device=self.device))
				loss += self.loss(transformed, candidate_embs, answers, ee_label).to(self.device)
				loss.backward()
				optimizer.step()
				tqdmloop.set_description("Epoch %d - Loss %.4f" % (epoch, float(loss)))
			if epoch % self.args.eval_per_epoch == 0:
				if dev_data is not None and epoch % self.args.eval_per_epoch == 0:
					with torch.no_grad():
						self.discovery_model.eval()
						self.transformer.eval()
						self.data.reset_new_entity()
						preds = []
						labels = []
						out_kb_flags = [x.is_new_entity for x in dev_data.eld_items]
						for batch in dev_batch:
							args, kwargs = self.prepare_input(batch)
							label = batch[-2]
							_, encoded_mention = self.discovery_model(*args, **kwargs)
							transformed = self.transformer(encoded_mention)
							preds.append(transformed)
							labels.append(label)
						preds = torch.cat(preds)
						idx, sims = self.data.predict_entity_with_embedding_train(dev_data.eld_items, preds, out_kb_flags)

						labels = torch.cat(labels)
						evals = {}
						for threshold_idx in range(1, 20):
							_, total_score, in_kb_score, out_kb_score, _, ari, mapping_result, _ = evaluator.evaluate(dev_data.eld_items, out_kb_flags, idx[threshold_idx], out_kb_flags, labels)
							evals[threshold_idx] = [total_score[0], in_kb_score[0], out_kb_score[0], ari, mapping_result]
						p, r, f = 0, 0, 0
						okp, okr, okf = 0, 0, 0
						mt = 0
						cl = 0
						for k, v in evals.items():
							kp, kr, kf = v[0]
							if kf > f:
								p, r, f = kp, kr, kf
								okp, okr, okf = v[2]
								mt = self.data.calc_threshold(k)
								cl = len(v[4])

						gl.logger.info("Epoch %d eval score: P %.2f R %.2f F %.2f @ threshold %.1f" % (epoch, p * 100, r * 100, f * 100, mt))
						gl.logger.info("%s Out-KB score: P %.2f R %.2f F %.2f @ threshold %.2f" % (self.model_name, okp * 100, okr * 100, okf * 100, mt))
						gl.logger.info("%s # of cluster: %d @ threshold %.2f" % (self.model_name, cl, mt))
						if f > max_score[-1]:
							max_score = (p, r, f)
							max_score_epoch = epoch
							max_threshold = mt
							self.args.new_ent_threshold = max_threshold
							self.save_model()
						p, r, f = max_score
						gl.logger.info("Max eval score @ epoch %d: P %.2f R %.2f F %.2f @ threshold %.1f" % (max_score_epoch, p * 100, r * 100, f * 100, max_threshold))
						if not os.path.isdir("runs/eld/%s" % self.model_name):
							os.mkdir("runs/eld/%s" % self.model_name)
						jsondump(self.generate_result_dict(dev_data.eld_items, idx, sims, evals), "runs/eld/%s/%s_%d.json" % (self.model_name, self.model_name, epoch))
						if self.args.early_stop <= epoch - max_score_epoch:
							break
		if test_data is not None:
			jsondump(self.test(test_data), "runs/eld/%s/%s_test.json" % (self.model_name, self.model_name))

	def test(self, test_data: Corpus, out_kb_flags=None):
		self.load_model()
		self.data.oe2i = {}
		with torch.no_grad():
			self.discovery_model.eval()
			self.transformer.eval()
			self.data.reset_new_entity()

			self.data.initialize_corpus_tensor(test_data, train=False)
			evaluator = Evaluator(ELDArgs(), self.data)
			test_dataset = ELDDataset(self.mode, test_data, self.args, cand_dict=self.data.surface_ent_dict, limit=self.args.train_corpus_limit)
			test_batch = DataLoader(dataset=test_dataset, batch_size=512, shuffle=False)
			preds = []
			labels = []
			out_kb_labels = [x.is_new_entity for x in test_data.eld_items]
			if out_kb_flags is None:
				out_kb_flags = out_kb_labels
			for batch in test_batch:
				args, kwargs = self.prepare_input(batch)
				label = batch[-2]
				_, encoded_mention = self.discovery_model(*args, **kwargs)
				transformed = self.transformer(encoded_mention)
				preds.append(transformed)
				labels.append(label)
			preds = torch.cat(preds)

			idx, sims = self.data.predict_entity_with_embedding_train(test_data.eld_items, preds, out_kb_flags)
			# for item in idx.values():
			# 	print(max(item) - len(self.data.original_e2i))
			labels = torch.cat(labels)
			evals = {}
			for threshold_idx in range(1, 20):
				_, total_score, in_kb_score, out_kb_score, _, ari, mapping_result, _ = evaluator.evaluate(test_data.eld_items, out_kb_flags, idx[threshold_idx], out_kb_labels, labels)
				evals[threshold_idx] = [total_score[0], in_kb_score[0], out_kb_score[0], ari, mapping_result]
			p, r, f = 0, 0, 0
			okp, okr, okf = 0, 0, 0
			cl = 0
			mt = 0
			for k, v in evals.items():
				kp, kr, kf = v[0]
				if kf > f:
					p, r, f = kp, kr, kf
					okp, okr, okf = v[2]
					cl = len(v[4])
					mt = self.data.calc_threshold(k)
			print()
			gl.logger.info("%s Test score: P %.2f R %.2f F %.2f @ threshold %.2f" % (self.model_name, p * 100, r * 100, f * 100, mt))
			gl.logger.info("%s Out-KB score: P %.2f R %.2f F %.2f @ threshold %.2f" % (self.model_name, okp * 100, okr * 100, okf * 100, mt))
			gl.logger.info("%s # of cluster: %d @ threshold %.2f" % (self.model_name, cl, mt))

			# jsondump(self.generate_result_dict(test_data.eld_items, idx, sims, evals), "runs/eld/%s/%s_test2.json" % (self.model_name, self.model_name))

			return self.generate_result_dict(test_data.eld_items, idx, sims, evals)

	def load_model(self):
		self.discovery_model: nn.Module = SeparateEntityEncoder(self.args)
		if self.args.vector_transformer == "cnn":
			self.transformer: nn.Module = CNNVectorTransformer(self.discovery_model.max_input_dim, self.args.e_emb_dim, self.args.flags)
		elif self.args.vector_transformer == "ffnn":
			self.transformer: nn.Module = FFNNVectorTransformer(self.discovery_model.max_input_dim, self.args.e_emb_dim, self.args.flags)

		d = torch.load(self.args.model_path) if self.mode != "demo" else torch.load(self.args.model_path, map_location=lambda storage, location: storage)
		self.discovery_model.load_state_dict(d["discovery"])
		self.transformer.load_state_dict(d["transformer"])
		self.discovery_model.to(self.device)

	def save_model(self):
		jsondump(self.args.to_json(), self.args.arg_path)
		d = {
			"discovery": self.discovery_model.state_dict(),
			"transformer": self.transformer.state_dict()
		}
		torch.save(d, self.args.model_path)

	def prepare_input(self, batch):
		if self.mode == "train":
			ce, cl, we, wl, lwe, lwl, rwe, rwl, lee, lel, ree, rel, re, rl, te, tl, in_cand_dict_flag, avg_degree = [x.to(self.device, torch.float32) if x is not None else None for x in batch[:-6]]
		elif self.mode == "test":
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

	def loss(self, pred_emb, candidates, answers, ee_label):
		assert pred_emb.size(0) == candidates.size(0) == answers.size(0)
		def get_pred(tensor, emb):
			assert emb.dim() == 2
			if emb.size(0) == 0: return 0, 0
			emb = emb.to(self.device)
			expanded = tensor.expand_as(emb)
			cos_sim = F.cosine_similarity(expanded, emb)
			# cos_sim = torch.bmm(emb.unsqueeze(1), expanded.unsqueeze(2)).squeeze()
			# dist = F.pairwise_distance(expanded, emb)
			# dist += 1 # prevent zero division
			return cos_sim

		softmax_loss = []

		for pe, cand in zip(pred_emb, candidates):
			sims = torch.softmax(get_pred(pe, cand.clone().detach()), 0)
			softmax_loss.append(sims)

		l1 = F.cross_entropy(torch.stack(softmax_loss).to(self.device), answers.to(self.device), reduction="mean")
		l2 = F.mse_loss(pred_emb.to(self.device), ee_label.to(self.device), reduction="sum") / pred_emb.size(0)
		return l1 + l2


