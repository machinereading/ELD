import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.ds import Corpus, CandDict
from .modules import CNNVectorTransformer, SeparateEntityEncoder
from .utils import ELDArgs, DataModule, Evaluator
from .utils.Dataset import ELDDataset, SkipgramDataset
from .. import GlobalValues as gl
from ..utils import jsondump

class SkipGramEntEmbedding:
	def __init__(self, mode: str, model_name: str, args: ELDArgs, data: DataModule = None):
		gl.logger.info("Initializing prediction model")
		self.mode = mode
		self.model_name = model_name
		self.args = args
		self.device = "cuda"
		if data is None:
			self.data = DataModule(mode, args)
		else:
			self.data = data
		if self.mode == "train":
			self.encoder: nn.Module = SeparateEntityEncoder(self.args)
			self.transformer: nn.Module = CNNVectorTransformer(self.encoder.max_input_dim, args.e_emb_dim, args.flags)

			self.word_post_transformer = nn.Embedding(len(self.data.w2i), args.e_emb_dim)
			self.entity_post_transformer_o = nn.Embedding(len(self.data.e2i), args.e_emb_dim)  # 주변개체 예측용
			self.entity_post_transformer_i = nn.Embedding(len(self.data.e2i), args.e_emb_dim)  # 중심개체 예측용

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

		self.train_method = args.pred_train_mode
		gl.logger.info("Finished prediction model initialization")

	def train(self, train_data: Corpus, dev_data: Corpus = None, test_data: Corpus = None):
		gl.logger.info("Training prediction model")
		evaluator = Evaluator(self.args, self.data)
		batch_size = 512
		optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.transformer.parameters()) + list(self.word_post_transformer.parameters()) + list(self.entity_post_transformer_o.parameters()), lr=1e-4, weight_decay=1e-4)
		tqdmloop = tqdm(range(1, self.args.epochs + 1))
		train_dataset = SkipgramDataset(self.mode, train_data, self.args, cand_dict=self.data.surface_ent_dict, limit=self.args.train_corpus_limit)
		train_batch = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
		self.args.jamo_limit = train_dataset.max_jamo_len_in_word
		self.args.word_limit = train_dataset.max_word_len_in_entity
		dev_batch = None
		if dev_data is not None:
			self.data.initialize_corpus_tensor(dev_data)
			dev_dataset = ELDDataset(self.mode, dev_data, self.args, cand_dict=self.data.surface_ent_dict, limit=self.args.train_corpus_limit)
			dev_batch = DataLoader(dataset=dev_dataset, batch_size=512, shuffle=False, num_workers=4)

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
				entity_idx = batch[-2].to(self.device)
				pos_word_sample, neg_word_sample, pos_ent_sample, neg_ent_sample = batch[-4:].to(self.device)
				# loss = sum([F.mse_loss(p, g) if torch.sum(g) != 0 else torch.zeros_like(F.mse_loss(p, g)) for p, g in zip(transformed, ee_label)])
				loss = self.skipgramloss(transformed, entity_idx, pos_word_sample, neg_word_sample, pos_ent_sample, neg_ent_sample)

				loss.backward()
				optimizer.step()
				tqdmloop.set_description("Epoch %d - Loss %.4f" % (epoch, float(loss)))

			if dev_data is not None and epoch % self.args.eval_per_epoch == 0:
				self.encoder.eval()
				self.transformer.eval()
				self.data.reset_new_entity()
				preds = []
				labels = []
				out_kb_flags = [x.is_new_entity for x in dev_data.eld_items]
				for batch in dev_batch:
					args, kwargs = self.prepare_input(batch)
					label = batch[-2]
					_, encoded_mention = self.encoder(*args, **kwargs)
					transformed = self.transformer(encoded_mention)
					preds.append(transformed)
					labels.append(label)
				preds = torch.cat(preds, dim=-1)
				pred_idx_gold_discovery = self.data.predict_entity_with_embedding_train(dev_data.eld_items, preds, out_kb_flags)

				labels = torch.cat(labels, dim=-1)
				_, total_score, in_kb_score, out_kb_score, _, ari, mapping_result, _ = evaluator.evaluate(dev_data.eld_items, out_kb_flags, None, pred_idx_gold_discovery, out_kb_flags, labels)

				p, r, f = total_score
				gl.logger.info("Epoch %d eval score: P %.2f R %.2f F %.2f" % (epoch, p * 100, r * 100, f * 100))
				if f > max_score[-1]:
					max_score = (p, r, f)
					max_score_epoch = epoch
					self.save_model()
				p, r, f = max_score
				gl.logger.info("Max eval score @ epoch %d: P %.2f R %.2f F %.2f" % (max_score_epoch, p * 100, r * 100, f * 100))
		np.save(self.model_name + "_entity_emb", self.entity_post_transformer_i.weight.numpy())  # 이게 진짜 entity embedding으로 동작할 수 있는지?
		if test_data is not None:
			self.data.initialize_corpus_tensor(test_data)
			test_dataset = ELDDataset(self.mode, test_data, self.args, cand_dict=self.data.surface_ent_dict, limit=self.args.train_corpus_limit)

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
			"encoder"                  : self.encoder.state_dict(),
			"transformer"              : self.transformer.state_dict(),
			"word_post_transformer"    : self.word_post_transformer.state_dict(),
			"entity_post_transformer_o": self.entity_post_transformer_o.state_dict(),
			"entity_post_transformer_i": self.entity_post_transformer_i.state_dict()
		}
		torch.save(save_dict, self.args.model_path)

	def skipgramloss(self, pred_emb, token_idx, token_pos_sample, token_neg_sample, entity_pos_sample, entity_neg_sample):
		# code from https://github.com/theeluwin/pytorch-sgns/blob/master/model.py
		# tps, tns, eps, ens = [x.transpose(0, 1).to(self.device) for x in (token_pos_sample, token_neg_sample, entity_pos_sample, entity_neg_sample)] # batch * sample_size
		tps, tns = [self.word_post_transformer(x) for x in [token_pos_sample, token_neg_sample]]  # batch * sample_size * e_emb_dim

		eps, ens = [self.entity_post_transformer_o(x) for x in [entity_pos_sample, entity_neg_sample]]  # batch * sample_size * e_emb_dim
		idx_enc = self.entity_post_transformer_i(token_idx)
		tns = tns.neg()
		ens = ens.neg()

		pred_emb = pred_emb.unsqueeze(2)
		entloss = torch.bmm(idx_enc.unsqueeze(0), pred_emb).squeeze().sigmoid().log().mean(1)
		tploss = torch.bmm(tps, pred_emb).squeeze().sigmoid().log().mean(1)
		eploss = torch.bmm(eps, pred_emb).squeeze().sigmoid().log().mean(1)
		tnloss = torch.bmm(tns, pred_emb).squeeze().sigmoid().log().view(-1, token_pos_sample.size(1), token_neg_sample.size(1)).sum(2).mean(1)
		enloss = torch.bmm(ens, pred_emb).squeeze().sigmoid().log().view(-1, entity_pos_sample.size(1), entity_neg_sample.size(1)).sum(2).mean(1)
		return -sum([entloss, tploss, tnloss, eploss, enloss]).mean()

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

class MSEEntEmbedding:
	def __init__(self, mode: str, model_name: str, args: ELDArgs=None, data: DataModule = None):
		gl.logger.info("Initializing prediction model")
		self.mode = mode
		self.model_name = model_name
		self.args = args
		self.device = "cuda" if self.mode != "demo" else "cpu"

		if self.mode == "train":
			self.encoder: nn.Module = SeparateEntityEncoder(self.args)
			self.transformer: nn.Module = CNNVectorTransformer(self.encoder.max_input_dim, args.e_emb_dim, args.flags)
			jsondump(args.to_json(), "models/eld/%s_args.json" % model_name)
		else:
			self.args = ELDArgs.from_json("models/eld/%s_args.json" % model_name)
			self.load_model()
		if data is None:
			self.data = DataModule(mode, self.args)
		else:
			self.data = data
		self.encoder.to(self.device)
		self.transformer.to(self.device)
		self.args.device = self.device

		gl.logger.info("Finished prediction model initialization")

	def train(self, train_data: Corpus, dev_data: Corpus = None, test_data: Corpus = None):
		gl.logger.info("Training prediction model")

		batch_size = 512
		optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.transformer.parameters()), lr=1e-4, weight_decay=1e-4)
		tqdmloop = tqdm(range(1, self.args.epochs + 1))
		self.data.initialize_corpus_tensor(train_data)
		train_dataset = ELDDataset(self.mode, train_data, self.args, cand_dict=self.data.surface_ent_dict, limit=self.args.train_corpus_limit)
		train_batch = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
		self.args.jamo_limit = train_dataset.max_jamo_len_in_word
		self.args.word_limit = train_dataset.max_word_len_in_entity
		dev_batch = None
		if dev_data is not None:
			self.data.initialize_corpus_tensor(dev_data, train=False)
			dev_dataset = ELDDataset(self.mode, dev_data, self.args, cand_dict=self.data.surface_ent_dict, limit=self.args.train_corpus_limit)
			dev_batch = DataLoader(dataset=dev_dataset, batch_size=512, shuffle=False, num_workers=4)

		evaluator = Evaluator(ELDArgs(), self.data)
		max_score = (0, 0, 0)
		max_score_epoch = 0
		max_threshold = 0
		for epoch in tqdmloop:
			self.encoder.train()
			self.transformer.train()
			# nel = []
			# gei = []
			# preds = []
			newent_flags = []
			entity_idxs = []
			preds = []
			losssum = 0
			for batch in train_batch:
				optimizer.zero_grad()
				ce, cl, we, wl, lwe, lwl, rwe, rwl, lee, lel, ree, rel, re, rl, te, tl, _, _, _ = [x.to(self.device, torch.float32) if x is not None else None for x in batch[:-4]]
				args = (ce, cl, we, wl, lwe, lwl, rwe, rwl, lee, lel, ree, rel, re, rl, te, tl)
				# args, kwargs = self.prepare_input(batch)
				_, encoded_mention = self.encoder(*args)
				transformed = self.transformer(encoded_mention)
				preds.append(transformed)
				newent_flags.append(batch[-4])
				ee_label = batch[-3].to(self.device)
				entity_idxs.append(batch[-2])

				loss = sum([F.mse_loss(p, g) if torch.sum(g) != 0 else torch.zeros_like(F.mse_loss(p, g)) for p, g in zip(transformed, ee_label)]) / ee_label.size(0)

				loss.backward()
				optimizer.step()
				losssum += float(loss)
				tqdmloop.set_description("Epoch %d - Loss %.4f" % (epoch, losssum))
			self.data.update_new_entity_embedding(train_dataset, torch.cat(newent_flags), torch.cat(entity_idxs), torch.cat(preds), epoch)
			if dev_data is not None and epoch % self.args.eval_per_epoch == 0:
				with torch.no_grad():
					self.encoder.eval()
					self.transformer.eval()
					self.data.reset_new_entity()
					preds = []
					labels = []
					out_kb_flags = [x.is_new_entity for x in dev_data.eld_items]
					for batch in dev_batch:
						args, kwargs = self.prepare_input(batch)
						label = batch[-2]
						_, encoded_mention = self.encoder(*args, **kwargs)
						transformed = self.transformer(encoded_mention)
						preds.append(transformed)
						labels.append(label)
					preds = torch.cat(preds)
					idx, sims = self.data.predict_entity_with_embedding_train(dev_data.eld_items, preds, out_kb_flags)

					labels = torch.cat(labels)
					evals = {}
					for threshold_idx in range(1, 10):
						_, total_score, in_kb_score, out_kb_score, _, ari, mapping_result, _ = evaluator.evaluate(dev_data.eld_items, out_kb_flags, idx[threshold_idx], out_kb_flags, labels)
						evals[threshold_idx] = [total_score[0], in_kb_score[0], out_kb_score[0], ari, mapping_result]
					p, r, f = 0, 0, 0
					mt = 0
					for k, v in evals.items():
						kp, kr, kf = v[0]
						if kf > f:
							p, r, f = kp, kr, kf
							mt = 0.5 + k * 0.02

					gl.logger.info("Epoch %d eval score: P %.2f R %.2f F %.2f @ threshold %.1f" % (epoch, p * 100, r * 100, f * 100, mt))
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
		with torch.no_grad():
			self.encoder.eval()
			self.transformer.eval()
			self.data.reset_new_entity()

			self.data.initialize_corpus_tensor(test_data)
			evaluator = Evaluator(ELDArgs(), self.data)
			test_dataset = ELDDataset(self.mode, test_data, self.args, cand_dict=self.data.surface_ent_dict, limit=self.args.train_corpus_limit)
			test_batch = DataLoader(dataset=test_dataset, batch_size=512, shuffle=False, num_workers=4)
			preds = []
			labels = []
			out_kb_labels = [x.is_new_entity for x in test_data.eld_items]
			if out_kb_flags is None:
				out_kb_flags = out_kb_labels
			for batch in test_batch:
				args, kwargs = self.prepare_input(batch)
				label = batch[-2]
				_, encoded_mention = self.encoder(*args, **kwargs)
				transformed = self.transformer(encoded_mention)
				preds.append(transformed)
				labels.append(label)
			preds = torch.cat(preds)

			idx, sims = self.data.predict_entity_with_embedding_train(test_data.eld_items, preds, out_kb_flags)
			print(max(*idx) - len(self.data.original_e2i))
			labels = torch.cat(labels)
			evals = {}
			for threshold_idx in range(1, 10):
				_, total_score, in_kb_score, out_kb_score, _, ari, mapping_result, _ = evaluator.evaluate(test_data.eld_items, out_kb_flags, idx[threshold_idx], out_kb_labels, labels)
				evals[threshold_idx] = [total_score[0], in_kb_score[0], out_kb_score[0], ari, mapping_result]
			p, r, f = 0, 0, 0
			mt = 0
			for k, v in evals.items():
				kp, kr, kf = v[0]
				if kf > f:
					p, r, f = kp, kr, kf
					mt = self.data.calc_threshold(k)
			gl.logger.info("Test score: P %.2f R %.2f F %.2f @ threshold %.2f" % (p * 100, r * 100, f * 100, mt))
			# jsondump(self.generate_result_dict(test_data.eld_items, idx, sims, evals), "runs/eld/%s/%s_test2.json" % (self.model_name, self.model_name))
			return self.generate_result_dict(test_data.eld_items, idx, sims, evals), "runs/eld/%s/%s_test2.json" % (self.model_name, self.model_name)
			# jsondump(self.args.to_json(), self.args.arg_path)

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
		if self.mode == "demo":
			load_dict = torch.load(self.args.model_path, map_location=lambda storage, location: storage)
		else:
			load_dict = torch.load(self.args.model_path)
		self.encoder.load_state_dict(load_dict["encoder"])
		self.transformer.load_state_dict(load_dict["transformer"])

		self.encoder.to(self.device)
		self.transformer.to(self.device)


	def save_model(self):
		save_dict = {
			"encoder"    : self.encoder.state_dict(),
			"transformer": self.transformer.state_dict()
		}
		torch.save(save_dict, self.args.model_path)
		jsondump(self.args.to_json(), self.args.arg_path)

	def predict(self, data: Corpus): # For API
		self.data.initialize_corpus_tensor(data, pred=True)
		dataset = ELDDataset(self.mode, data, self.args, cand_dict=self.data.surface_ent_dict, limit=self.args.train_corpus_limit)
		batchs = DataLoader(dataset=dataset, batch_size=512, shuffle=False, num_workers=4)
		preds = []
		labels = []
		out_kb_flags = [x.is_dark_entity for x in data.entities]
		for batch in batchs:
			args, kwargs = self.prepare_input(batch)
			label = batch[-2]
			_, encoded_mention = self.encoder(*args, **kwargs)
			transformed = self.transformer(encoded_mention)
			preds.append(transformed)
			labels.append(label)
		preds = torch.cat(preds)

		result, sims = self.data.predict_entity_with_embedding_immediate(data.entities, preds, out_kb_flags)
		for e, r, s in zip(data.entities, result, sims):
			if e.entity == "NOT_IN_CANDIDATE":
				e.entity = r
			e.confidence_score = s
		return data

	def __call__(self, data: Corpus):
		return self.predict(data)

	def generate_result_dict(self, eld_items, preds, sims, eval_result):
		result = {
			"score": {},
			"data" : []
		}
		for k in eval_result.keys():
			v = eval_result[k]
			pred = preds[k]
			sim = sims[k]
			result["score"][k] = {
				"Total" : list(v[0]),
				"In-KB" : list(v[1]),
				"Out-KB": list(v[2]),
				"ARI"   : v[3]
			}
			mapping_result = v[4]
			for i, (e, p, s) in enumerate(zip(eld_items, pred, sim)):
				if p >= self.data.original_entity_embedding.size(0):
					preassigned = mapping_result[p] < 0
					mapping = mapping_result[p] if not preassigned else mapping_result[p] * -1

					p = self.data.i2oe[mapping - len(self.data.e2i)]
					if preassigned:
						p += "_preassigned"
				else:
					p = self.data.i2e[p]
				if len(result["data"]) <= i:
					result["data"].append({
						"Surface": e.surface,
						"Context": " ".join([x.surface for x in e.lctx[-5:]] + ["[%s]" % e.surface] + [x.surface for x in e.rctx[:5]]),
						"EntPred": {k: "%s:%.2f" % (p, s)},
						"Entity" : e.entity
					})
				else:
					result["data"][i]["EntPred"][k] = "%s:%.2f" % (p, s)
		return result

class DictBasedPred:
	"""
	개체 연결 -> 실패 -> 등록 -> 재연결 시도
	"""

	def __init__(self, data: DataModule):
		self.data = data

		self.canddict = data.surface_ent_dict
		self.model_name = "dictbased"

	def test(self, data: Corpus):
		self.data.initialize_corpus_tensor(data, pred=True)
		evaluator = Evaluator(ELDArgs(), self.data)
		preds = []
		labels = []
		out_kb_flags = [x.is_new_entity for x in data.eld_items]
		out_kb_preds = []
		for entity in data.eld_items:
			cands = self.canddict[entity.surface]
			if len(cands) == 0:
				ent = len(self.data.e2i)
				self.canddict.add_instance(entity.surface, ent)
				self.data.e2i[len(self.data.e2i)] = ent
			else:
				ent = self.data.e2i[cands[0][0]] if cands[0][0] in self.data.e2i else 0
			preds.append(ent)
			out_kb_preds.append(ent >= len(self.data.original_e2i))
			labels.append(self.data.e2i[entity.entity] if entity.entity in self.data.e2i else 0)

		_, total_score, in_kb_score, out_kb_score, _, ari, mapping_result, _ = evaluator.evaluate(data.eld_items, out_kb_preds, preds, out_kb_flags, labels)
		print(total_score[0])
		return self.generate_result_dict(data.eld_items, preds, (total_score, in_kb_score, out_kb_score, ari, mapping_result))

	def generate_result_dict(self, eld_items, preds, eval_result):
		result = {
			"score": {},
			"data" : []
		}
		v = eval_result
		pred = preds
		result["score"] = {
			"Total" : list(v[0]),
			"In-KB" : list(v[1]),
			"Out-KB": list(v[2]),
			"ARI"   : v[3]
		}
		mapping_result = v[4]
		for i, (e, p) in enumerate(zip(eld_items, pred)):
			if p >= self.data.original_entity_embedding.size(0):
				preassigned = mapping_result[p] < 0
				mapping = mapping_result[p] if not preassigned else mapping_result[p] * -1
				if mapping == 0:
					p = "NOT_IN_CANDIDATE"
				else:
					if mapping < len(self.data.original_e2i): p = "NOT_IN_CANDIDATE"
					else: p = self.data.i2oe[mapping - len(self.data.original_e2i)]
				if preassigned:
					p += "_preassigned"
			else:
				p = self.data.i2e[p]
			result["data"].append({
				"Surface": e.surface,
				"Context": " ".join([x.surface for x in e.lctx[-5:]] + ["[%s]" % e.surface] + [x.surface for x in e.rctx[:5]]),
				"EntPred": p,
				"Entity" : e.entity
			})
		return result

class NoRegister(MSEEntEmbedding):
	def __init__(self, mode: str, model_name: str, args: ELDArgs=None, data: DataModule = None):
		super(NoRegister, self).__init__(mode, model_name, args, data)
		self.model_name = "noreg"

	def test(self, test_data: Corpus):
		self.load_model()
		with torch.no_grad():
			self.encoder.eval()
			self.transformer.eval()
			self.data.reset_new_entity()

			self.data.initialize_corpus_tensor(test_data)
			evaluator = Evaluator(ELDArgs(), self.data)
			test_dataset = ELDDataset(self.mode, test_data, self.args, cand_dict=self.data.surface_ent_dict, limit=self.args.train_corpus_limit)
			test_batch = DataLoader(dataset=test_dataset, batch_size=512, shuffle=False, num_workers=4)
			preds = []
			labels = []
			out_kb_labels = [x.is_new_entity for x in test_data.eld_items]
			out_kb_preds = [0 for _ in test_data.eld_items]
			for batch in test_batch:
				args, kwargs = self.prepare_input(batch)
				label = batch[-2]
				_, encoded_mention = self.encoder(*args, **kwargs)
				transformed = self.transformer(encoded_mention)
				preds.append(transformed)
				labels.append(label)
			preds = torch.cat(preds)

			idx, sims = self.data.predict_entity_with_embedding_train(test_data.eld_items, preds, out_kb_preds)

			labels = torch.cat(labels)
			_, total_score, in_kb_score, out_kb_score, _, ari, mapping_result, _ = evaluator.evaluate(test_data.eld_items, out_kb_preds, idx[1], out_kb_labels, labels)
			evals = [total_score[0], in_kb_score[0], out_kb_score[0], ari, mapping_result]
			p, r, f = total_score[0]
			gl.logger.info("Test score: P %.2f R %.2f F %.2f" % (p * 100, r * 100, f * 100))
			# jsondump(self.generate_result_dict(test_data.eld_items, idx, sims, evals), "runs/eld/%s/%s_test2.json" % (self.model_name, self.model_name))
			return self.generate_result_dict(test_data.eld_items, idx, evals)

	def generate_result_dict(self, eld_items, preds, eval_result):
		result = {
			"score": {},
			"data" : []
		}
		v = eval_result
		pred = preds
		result["score"] = {
			"Total" : list(v[0]),
			"In-KB" : list(v[1]),
			"Out-KB": list(v[2]),
			"ARI"   : v[3]
		}
		mapping_result = v[4]
		for i, (e, p) in enumerate(zip(eld_items, pred)):
			if p >= self.data.original_entity_embedding.size(0):
				preassigned = mapping_result[p] < 0
				mapping = mapping_result[p] if not preassigned else mapping_result[p] * -1

				p = self.data.i2oe[mapping - len(self.data.e2i)]
				if preassigned:
					p += "_preassigned"
			else:
				p = self.data.i2e[p]
			result["data"].append({
				"Surface": e.surface,
				"Context": " ".join([x.surface for x in e.lctx[-5:]] + ["[%s]" % e.surface] + [x.surface for x in e.rctx[:5]]),
				"EntPred": p,
				"Entity" : e.entity
			})
		return result
