import os
import random
import traceback
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

from src.ds import Corpus
from .modules import SeparateEntityEncoder, FFNNEncoder, FFNNVectorTransformer, SelfAttnVectorTransformer, CNNVectorTransformer
from .utils import ELDArgs, DataModule, Evaluator, TypeEvaluator
from .. import GlobalValues as gl
from ..utils import jsondump, split_to_batch

# from tabulate import tabulate

# noinspection PyMethodMayBeStatic
class ELDSkeleton(ABC):
	def __init__(self, mode: str, model_name: str, train_new=True, train_args: ELDArgs = None):
		self.model_name = model_name
		self.mode = mode
		self.device = "cuda" if torch.cuda.is_available() and mode != "demo" else "cpu"
		self.is_best_model = False
		self.args = None
		if mode not in ["train", "typeeval"]:
			gl.logger.info("Loading model")
			self.load_model()
			self.is_best_model = True
			gl.logger.info("Loading finished")
		else:
			try:
				if train_new:
					if train_args is not None:
						self.args = train_args
					else:
						self.args = ELDArgs(model_name)
				else:
					self.args = ELDArgs.from_json("models/eld/%s_args.json" % model_name)
					self.load_model()
			except Exception:
				traceback.print_exc()
				self.args = ELDArgs(model_name)
			self.epochs = self.args.epochs
			self.eval_per_epoch = self.args.eval_per_epoch
			self.model_path = self.args.model_path
		if self.args is None:
			self.args = ELDArgs() if train_args is None else train_args

		self.args.device = self.device
		self.args.mode = mode
		self.data = DataModule(mode, self.args)
		if self.args.type_prediction:
			self.type_evaluator = TypeEvaluator()
			if mode == "typeeval": return
		if mode in ["pred", "demo"]: return
		self.evaluator = Evaluator(self.args, self.data)

		self.stop = self.args.early_stop
		self.stop_train_discovery = False

	def discovery_loss(self, kb_score, new_entity_flag):
		return F.binary_cross_entropy_with_logits(kb_score.view(-1), new_entity_flag.float())

	def out_kb_loss(self, pred, new_entity_flag, label):
		l1 = sum([F.mse_loss(p, g) if t == 0 else torch.zeros_like(F.mse_loss(p, g)) for t, p, g in zip(new_entity_flag, pred, label)])
		l2 = sum([F.mse_loss(p, g) if t == 1 and torch.sum(g) != 0 else torch.zeros_like(F.mse_loss(p, g)) for t, p, g in zip(new_entity_flag, pred, label)])  # zero tensor는 일단 거름
		# return sum([F.mse_loss(p, g) for t, p, g in zip(new_entity_flag, pred, label) if torch.sum(g) != 0])
		return l1 + l2

	def posteval(self, epoch, max_score_epoch, max_score, dev_corpus, new_ent_preds, pred_entity_idxs, new_entity_labels, gold_entity_idxs):
		kb_expectation_score, total_score, in_kb_score, out_kb_score, no_surface_score, cluster_score, mapping_result_clustered, mapping_result_unclustered = self.evaluator.evaluate(dev_corpus, torch.tensor(new_ent_preds).view(-1),
		                                                                                                                                                                              torch.tensor(pred_entity_idxs),
		                                                                                                                                                                              torch.cat(new_entity_labels, dim=-1).cpu(),
		                                                                                                                                                                              torch.cat(gold_entity_idxs, dim=-1).cpu())
		print()
		p, r, f = kb_expectation_score
		gl.logger.info("%s score: P %.2f, R %.2f, F1 %.2f" % ("KB Expectation", p * 100, r * 100, f * 100))
		for score_info, ((cp, cr, cf), (up, ur, uf)) in [["Total", total_score],
		                                                 ["in-KB", in_kb_score],
		                                                 ["out KB", out_kb_score],
		                                                 ["No surface", no_surface_score]]:
			gl.logger.info("%s score: Clustered - P %.2f, R %.2f, F1 %.2f, Unclustered - P %.2f, R %.2f, F1 %.2f" % (score_info, cp * 100, cr * 100, cf * 100, up * 100, ur * 100, uf * 100))
		gl.logger.info("Clustering score: %.2f" % (cluster_score * 100))
		if total_score[0][-1] > max_score:
			max_score = total_score[0][-1]
			max_score_epoch = epoch
			self.save_model()
			self.is_best_model = True
		gl.logger.info("Best epoch %d - Score %.2f" % (max_score_epoch, max_score * 100))
		if not os.path.isdir("runs/eld/%s" % self.model_name):
			os.mkdir("runs/eld/%s" % self.model_name)
		analyze_data = self.data.analyze(dev_corpus, torch.tensor(new_ent_preds).view(-1), torch.tensor(pred_entity_idxs), torch.cat(new_entity_labels).cpu(), torch.cat(gold_entity_idxs).cpu(),
		                                 (kb_expectation_score, total_score, in_kb_score, out_kb_score, no_surface_score, cluster_score, mapping_result_clustered, mapping_result_unclustered))
		jsondump(analyze_data, "runs/eld/%s/%s_%d.json" % (self.model_name, self.model_name, epoch))
		if kb_expectation_score[-1] > 0.8 and not self.stop_train_discovery:
			self.stop_train_discovery = True
			print("Stop train discovery")
		if epoch - max_score_epoch > self.stop:
			gl.logger.info("No better performance for %d epoch - Training stop" % self.stop)
			return False, max_score_epoch, max_score, analyze_data
		return True, max_score_epoch, max_score, analyze_data

	@abstractmethod
	def save_model(self):
		pass

	@abstractmethod
	def load_model(self):
		pass

	@abstractmethod
	def predict(self, *data):
		pass

	def __call__(self, *data):
		if self.mode == "demo":
			data = data[0]["content"]
			result = self.predict(data)
			return self.postprocess(result)
		else:
			result = self.predict(*data)
			return result


	def evaluate_type(self):  # hard-coded for type prediction # TODO Temporary code
		if not self.args.type_prediction: return
		preds = self.data.typegiver(*self.data.corpus.eld_items)
		labels = self.data.typegiver.get_gold(*self.data.corpus.eld_items)
		print(TypeEvaluator()(self.data.corpus.eld_items, preds, labels))

	def postprocess(self, corpus):
		from ..demo.postprocess import postprocess
		# formatting
		result = []
		for sentence in corpus:
			buf = {
				"text"       : sentence.original_sentence,
				"entities"   : [],
				"dark_entity": []
			}
			for entity in sentence.entities:
				entity = postprocess(entity)
				voca = {
					"text": entity.surface,
					"start_offset": entity.char_ind,
					"end_offset": entity.char_ind + len(entity.surface),
					"ne_type": entity.ne_type,
					"type": entity.type,
					"score": 0,
					"confidence": entity.confidence_score,
					"uri": "http://kbox.kaist.ac.kr/resource/" + entity.entity,
					"en_entity": entity.en_entity
				}
				if entity.is_dark_entity:
					buf["dark_entity"].append(voca)
				else:
					buf["entities"].append(voca)
			result.append(buf)
		return result
# noinspection PyUnresolvedReferences
class VectorBasedELD(ELDSkeleton):
	def __init__(self, mode: str, model_name: str, train_new=True, train_args: ELDArgs = None):
		super(VectorBasedELD, self).__init__(mode, model_name, train_new, train_args)
		args = self.args
		if mode == "typeeval":
			return
		# if mode != "train":
		# 	self.transformer = SeparateEntityEncoder \
		# 		(args.use_character_embedding, args.use_word_embedding, args.use_word_context_embedding, args.use_entity_context_embedding, args.use_relation_embedding, args.use_type_embedding,
		# 		 args.character_encoder, args.word_encoder, args.word_context_encoder, args.entity_context_encoder, args.relation_encoder, args.type_encoder,
		# 		 args.c_emb_dim, args.w_emb_dim, args.e_emb_dim, args.r_emb_dim, args.t_emb_dim,
		# 		 args.c_enc_dim, args.w_enc_dim, args.wc_enc_dim, args.ec_enc_dim, args.r_enc_dim, args.t_enc_dim,
		# 		 args.jamo_limit, args.word_limit, args.relation_limit).to(self.device)
		# 	try:
		# 		self.transformer.load_state_dict(torch.load(args.model_path))
		# 	except:
		# 		gl.logger.critical("No model exists!")
		# 	self.modify_entity_embedding = args.modify_entity_embedding
		# 	self.modify_entity_embedding_weight = args.modify_entity_embedding_weight
		self.entity_index = {}
		self.i2e = {v: k for k, v in self.entity_index.items()}
		self.use_explicit_kb_classifier = args.use_explicit_kb_classifier
		self.train_embedding = args.train_embedding
		# if mode == "test":
		# 	self.entity_embedding = self.data.entity_embedding.weight
		# 	self.entity_embedding_dim = self.entity_embedding.size()[-1]
		self.map_threshold = args.out_kb_threshold
		if mode == "train":
			self.transformer = SeparateEntityEncoder(args).to(self.device)
			if args.use_separate_feature_encoder:
				self.transformer2 = SeparateEntityEncoder(args).to(self.device)
			mapping = {"cnn": CNNVectorTransformer, "ffnn": FFNNVectorTransformer, "attn": SelfAttnVectorTransformer}
			self.vector_transformer = mapping[self.args.vector_transformer](self.transformer.max_input_dim, args.e_emb_dim, args.flags).to(self.device)
			jsondump(args.to_json(), "models/eld/%s_args.json" % model_name)
			gl.logger.info("ELD Model load complete")

	def train(self):
		gl.logger.info("Train start")
		batch_size = 256
		train_batch = DataLoader(dataset=self.data.train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
		dev_batch = DataLoader(dataset=self.data.dev_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

		dev_corpus = self.data.dev_dataset
		tqdmloop = tqdm(range(1, self.epochs + 1))
		discovery_optimizer = torch.optim.Adam(self.transformer.parameters(), lr=1e-3, weight_decay=1e-4)
		tensor_params = list(self.vector_transformer.parameters())
		if self.args.use_separate_feature_encoder:
			tensor_params += list(self.transformer2.parameters())
		tensor_optimizer = torch.optim.Adam(tensor_params, lr=1e-3)
		# params = list(self.vector_transformer.parameters()) + list(self.transformer.parameters())
		# optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)
		# optimizer = torch.optim.SGD(params, lr=0.001, weight_decay=1e-4)
		max_score = 0
		max_score_epoch = 0
		analysis = {}
		for epoch in tqdmloop:
			self.transformer.train()
			self.vector_transformer.train()
			if self.args.use_separate_feature_encoder:
				self.transformer2.train()
			ne = []
			ei = []
			pr = []
			for batch in train_batch:
				discovery_optimizer.zero_grad()
				tensor_optimizer.zero_grad()
				# optimizer.zero_grad()
				ce, cl, we, wl, lwe, lwl, rwe, rwl, lee, lel, ree, rel, re, rl, te, tl, in_cand_flag, cand_emb = [x.to(self.device, torch.float) if x is not None else None for x in batch[:-4]]
				new_entity_label, ee_label, gold_entity_idx = [x.to(self.device) for x in batch[-4:-1]]
				kb_score, pred = self.transformer(ce, cl, we, wl, lwe, lwl, rwe, rwl, lee, lel, ree, rel, re, rl, te, tl)
				discovery_loss_val = torch.tensor(0)
				if not self.stop_train_discovery:
					discovery_loss_val = self.discovery_loss(kb_score, new_entity_label)
					discovery_loss_val.backward()
					discovery_optimizer.step()
				if self.args.use_separate_feature_encoder:
					_, pred = self.transformer2(ce, cl, we, wl, lwe, lwl, rwe, rwl, lee, lel, ree, rel, re, rl, te, tl)
					pred = self.vector_transformer(pred)
				else:
					pred = self.vector_transformer(pred.detach())
				tensor_loss_val = self.out_kb_loss(pred, new_entity_label, ee_label)
				# loss = discovery_loss_val + tensor_loss_val
				# loss.backward()
				# optimizer.step()
				tensor_loss_val.backward()
				tensor_optimizer.step()
				tqdmloop.set_description("Epoch %d: Loss %.4f" % (epoch, float(discovery_loss_val + tensor_loss_val)))
				ne.append(new_entity_label)
				ei.append(gold_entity_idx)
				pr.append(pred)
			self.data.update_new_entity_embedding(torch.cat(ne), torch.cat(ei), torch.cat(pr), epoch)
			self.is_best_model = False
			if epoch % self.eval_per_epoch == 0:
				new_ent_preds, pred_entity_idxs, new_entity_labels, gold_entity_idxs = self.eval(dev_corpus, dev_batch)

				run, max_score_epoch, max_score, analysis = self.posteval(epoch, max_score_epoch, max_score, dev_corpus.eld_items, new_ent_preds, pred_entity_idxs, new_entity_labels, gold_entity_idxs)
				if not run:
					jsondump(analysis, "runs/eld/%s/%s_best_eval.json" % (self.model_name, self.model_name))
					break

		else:
			analysis["epoch"] = max_score_epoch
			jsondump(analysis, "runs/eld/%s/%s_best.json" % (self.model_name, self.model_name))
		# test
		self.test()

	# def register_new_entity(self, surface, entity_embedding):
	# 	idx = len(self.entity_index)
	# 	register_form = "__" + surface.replace(" ", "_")
	# 	self.entity_index[register_form] = idx
	# 	self.i2e[idx] = register_form
	# 	self.entity_embedding = torch.cat((self.entity_embedding, entity_embedding.unsqueeze(0)), 0)
	# 	return register_form

	def save_model(self):
		d = {
			"transformer": self.transformer.state_dict(),
			"vector"     : self.vector_transformer.state_dict()
		}
		if self.args.use_separate_feature_encoder:
			d["transformer2"] = self.transformer2.state_dict()
		torch.save(d, self.model_path)
		gl.logger.info("Model saved")

	def load_model(self):
		self.args = ELDArgs.from_json("models/eld/%s_args.json" % self.model_name)
		args = self.args
		self.transformer = SeparateEntityEncoder(args).to(self.device)
		mapping = {"cnn": CNNVectorTransformer, "ffnn": FFNNVectorTransformer, "attn": SelfAttnVectorTransformer}
		self.vector_transformer = mapping[self.args.vector_transformer](self.transformer.max_input_dim, args.e_emb_dim, args.flags).to(self.device)
		if self.args.use_separate_feature_encoder:
			self.transformer2 = SeparateEntityEncoder(args).to(self.device)
		try:
			load = torch.load(args.model_path)
			self.transformer.load_state_dict(load["transformer"])
			self.vector_transformer.load_state_dict(load["vector"])
			if self.args.use_separate_feature_encoder: self.transformer2.load_state_dict(load["transformer2"])
		except:
			traceback.print_exc()
			gl.logger.critical("No model exists!")

	def eval(self, corpus, dataset):
		self.transformer.eval()
		self.vector_transformer.eval()
		if self.args.use_separate_feature_encoder:
			self.transformer2.eval()
		self.data.reset_new_entity()  # registered entity를 전부 지우고 다시 처음부터 등록 시퀀스 시작
		with torch.no_grad():
			new_entity_labels = []
			gold_entity_idxs = []
			kb_scores = []
			preds = []
			dev_idxs = []
			for batch in dataset:
				ce, cl, we, wl, lwe, lwl, rwe, rwl, lee, lel, ree, rel, re, rl, te, tl, in_cand_flag, cand_emb = [x.to(self.device, torch.float32) if x is not None else None for x in batch[:-4]]
				kb_score, pred = self.transformer(ce, cl, we, wl, lwe, lwl, rwe, rwl, lee, lel, ree, rel, re, rl, te, tl)
				kb_score = torch.sigmoid(kb_score)
				# print(pred)
				pred = self.vector_transformer(pred.detach(), eval=False)  # TODO why all same tensor?
				# print(pred)
				new_entity_label, _, gold_entity_idx = [x.to(self.device) for x in batch[-4:-1]]
				dev_idxs.append(batch[-1])

				kb_scores.append(kb_score)
				preds.append(pred)
				new_entity_labels.append(new_entity_label)
				gold_entity_idxs.append(gold_entity_idx)
			for l_s, i_s in zip(gold_entity_idxs, dev_idxs):
				for l, i in zip(l_s, i_s):
					v = corpus.eld_items[i]
					assert v.entity_label_idx == l
				assert v.entity_label_idx == l, "%d/%d/%s" % (l, v.entity_label_idx, v.entity)
			new_ent_preds, pred_entity_idxs = self.data.predict_entity(corpus.eld_items, torch.cat(kb_scores), torch.cat(preds))
		return new_ent_preds, pred_entity_idxs, new_entity_labels, gold_entity_idxs

	def test(self, batch_size=512):
		if not self.is_best_model:
			self.load_model()  # load best
		test_corpus = self.data.test_dataset
		test_batch = DataLoader(dataset=test_corpus, batch_size=batch_size, shuffle=False, num_workers=8)
		new_ent_preds, pred_entity_idxs, new_entity_labels, gold_entity_idxs = self.eval(test_corpus, test_batch)
		kb_expectation_score, total_score, in_kb_score, out_kb_score, no_surface_score, cluster_score, mapping_result_clustered, mapping_result_unclustered = self.evaluator.evaluate(test_corpus.eld_items,
		                                                                                                                                                                              torch.tensor(new_ent_preds).view(-1),
		                                                                                                                                                                              torch.tensor(pred_entity_idxs),
		                                                                                                                                                                              torch.cat(new_entity_labels, dim=-1).cpu(),
		                                                                                                                                                                              torch.cat(gold_entity_idxs, dim=-1).cpu())
		print()
		p, r, f = kb_expectation_score
		gl.logger.info("Test score")
		gl.logger.info("%s score: P %.2f, R %.2f, F1 %.2f" % ("KB Expectation", p * 100, r * 100, f * 100))
		for score_info, ((cp, cr, cf), (up, ur, uf)) in [["Total", total_score],
		                                                 ["in-KB", in_kb_score],
		                                                 ["out KB", out_kb_score],
		                                                 ["No surface", no_surface_score]]:
			gl.logger.info("%s score: Clustered - P %.2f, R %.2f, F1 %.2f, Unclustered - P %.2f, R %.2f, F1 %.2f" % (score_info, cp * 100, cr * 100, cf * 100, up * 100, ur * 100, uf * 100))
		gl.logger.info("Clustering score: %.2f" % (cluster_score * 100))
		analyze_data = self.data.analyze(test_corpus.eld_items, torch.tensor(new_ent_preds).view(-1), torch.tensor(pred_entity_idxs), torch.cat(new_entity_labels).cpu(), torch.cat(gold_entity_idxs).cpu(),
		                                 (kb_expectation_score, total_score, in_kb_score, out_kb_score, no_surface_score, cluster_score, mapping_result_clustered, mapping_result_unclustered))
		jsondump(analyze_data, "runs/eld/%s/%s_test.json" % (self.model_name, self.model_name))

	def predict(self, *data, batch_size=512):
		self.transformer.eval()
		self.vector_transformer.eval()
		if self.args.use_separate_feature_encoder:
			self.transformer2.eval()

		dataset = self.data.prepare("pred", *data)
		data = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=8)

		with torch.no_grad():
			kb_scores = []
			preds = []
			for batch in data:
				ce, cl, we, wl, lwe, lwl, rwe, rwl, lee, lel, ree, rel, re, rl, te, tl, in_cand_flag, cand_emb = [x.to(self.device, torch.float32) if x is not None else None for x in batch]
				kb_score, pred = self.transformer(ce, cl, we, wl, lwe, lwl, rwe, rwl, lee, lel, ree, rel, re, rl, te, tl)
				kb_score = torch.sigmoid(kb_score)
				if self.args.use_separate_feature_encoder:
					_, pred = self.transformer2(ce, cl, we, wl, lwe, lwl, rwe, rwl, lee, lel, ree, rel, re, rl, te, tl)
				pred = self.vector_transformer(pred)
				kb_scores.append(kb_score)
				preds.append(pred)
			new_ent_preds, pred_entity_idxs = self.data.predict_entity(dataset.eld_items, torch.cat(kb_scores), torch.cat(preds), output_as_idx=False)
		result = []
		for v, n, p in zip(dataset.eld_items, new_ent_preds, pred_entity_idxs):
			result.append([v.surface, v.entity, n, p])
		return result

# noinspection PyMethodMayBeStatic
class BertBasedELD(ELDSkeleton):
	def predict(self, data):
		pass

	def __init__(self, mode: str, model_name: str, train_new=True, train_args: ELDArgs = None):
		super(BertBasedELD, self).__init__(mode, model_name, train_new, train_args)
		pretrained_weight = "bert-base-multilingual-cased"
		self.tokenizer = BertTokenizer.from_pretrained(pretrained_weight, do_lower_case=False)
		self.tokenizer.add_special_tokens({"cls_token": "[CLS]", "additional_special_tokens": ["[e]", "[/e]"], "sep_token": "[SEP]"})
		self.transformer = BertModel.from_pretrained(pretrained_weight).to(self.device)
		self.transformer.resize_token_embeddings(len(self.tokenizer))
		self.binary_classifier = FFNNEncoder(768, 1, 768, 2).to(self.device)
		self.vector_transformer = FFNNEncoder(768, 300, 768, 2).to(self.device)

	def initialize_dataset(self, corpus):
		s = []
		pad = lambda tensor, size: F.pad(tensor, [0, size - tensor.size(0)])
		for item in corpus.eld_items:
			parent_sentence = item.parent_sentence
			marked_sentence = "[CLS] " + parent_sentence.original_sentence[:item.char_ind] + " [e] " + item.surface + " [/e] " + parent_sentence.original_sentence[item.char_ind + len(item.surface):]
			new_entity_label = item.is_new_entity
			target_embedding = item.entity_label_embedding
			gold_entity_idx = item.entity_label_idx
			encoded_sentence = pad(self.tokenizer.encode(marked_sentence, max_length=512, return_tensors="pt", add_special_tokens=True).squeeze(0), 512)
			s.append((encoded_sentence, new_entity_label, target_embedding, gold_entity_idx))
		return s

	def train(self):
		gl.logger.info("Train start")
		batch_size = 4
		train_corpus = self.data.train_dataset
		train_set = self.initialize_dataset(train_corpus)
		dev_corpus = self.data.dev_dataset
		dev_set = self.initialize_dataset(dev_corpus)

		tqdmloop = tqdm(range(1, self.epochs + 1))
		# discovery_optimizer = torch.optim.Adam(self.transformer.parameters(), lr=1e-3, weight_decay=1e-4)
		# tensor_optimizer = torch.optim.Adam(self.vector_transformer.parameters(), lr=1e-3)
		params = list(self.binary_classifier.parameters()) + list(self.vector_transformer.parameters()) + list(self.transformer.parameters())
		optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)
		# optimizer = torch.optim.SGD(params, lr=0.001, weight_decay=1e-4)
		max_score = 0
		max_score_epoch = 0
		analysis = {}

		for epoch in tqdmloop:
			self.transformer.train()
			self.binary_classifier.train()
			self.vector_transformer.train()
			optimizer.zero_grad()
			random.shuffle(train_set)
			ne = []
			ei = []
			pr = []
			for batch in split_to_batch(train_set, batch_size):
				encoded_sentence, in_kb_label, target_embedding, gold_entity_idx = zip(*batch)
				encoded_sequence = torch.stack(encoded_sentence).to(self.device)
				# print(encoded_sequence[:][:5])
				# print("encoded_sequence", encoded_sequence.size())
				# attention_mask = torch.where(encoded_sequence > 0, torch.ones_like(encoded_sequence), torch.zeros_like(encoded_sequence))
				# transformer_input = torch.stack(encoded_sequence).to(self.device)

				new_entity_label = torch.ByteTensor(in_kb_label).to(self.device)
				target_embedding = torch.stack(target_embedding).to(self.device)
				transformer_output = self.transformer(encoded_sequence)[0][:, 0, :]
				# print("transformer_output", transformer_output.size())
				kb_score = self.binary_classifier(transformer_output)
				# print("kb_score, new_ent", kb_score.size(), new_entity_label.size())
				discovery_loss_val = self.discovery_loss(kb_score, new_entity_label)

				pred = self.vector_transformer(transformer_output)
				tensor_loss_val = self.out_kb_loss(pred, new_entity_label, target_embedding)
				total_loss = discovery_loss_val + tensor_loss_val
				total_loss.backward()

				optimizer.step()
				ne.append(new_entity_label)
				ei.append(torch.LongTensor(gold_entity_idx))
				pr.append(pred)
			self.data.update_new_entity_embedding(torch.cat(ne), torch.cat(ei), torch.cat(pr))
			if epoch % self.eval_per_epoch == 0:
				with torch.no_grad():
					self.transformer.eval()
					self.binary_classifier.eval()
					self.vector_transformer.eval()
					self.data.reset_new_entity()
					# new_ent_preds = []
					# pred_entity_idxs = []
					# new_entity_labels = []
					# gold_entity_idxs = []
					# dev_batch_start_idx = 0
					# for batch in split_to_batch(dev_set, batch_size):
					# 	encoded_sentence, in_kb_label, target_embedding, gold_entity_idx = zip(*batch)
					# 	encoded_sequence = torch.stack(encoded_sentence).to(self.device)
					# 	# attention_mask = torch.where(encoded_sequence > 0, torch.ones_like(encoded_sequence), torch.zeros_like(encoded_sequence))
					#
					# 	new_entity_label = torch.ByteTensor(in_kb_label).to(self.device)
					# 	# print(new_entity_label.size())
					# 	transformer_output = self.transformer(encoded_sequence)[0][:, 0, :]
					# 	kb_score = self.binary_classifier(transformer_output)
					#
					# 	pred = self.vector_transformer(transformer_output)
					# 	new_ent_pred, entity_idx = self.data.predict_entity(kb_score, pred, dev_corpus.eld_get_item(slice(dev_batch_start_idx, dev_batch_start_idx + batch_size)))
					# 	new_ent_preds += new_ent_pred
					# 	pred_entity_idxs += entity_idx
					# 	new_entity_labels.append(new_entity_label)
					# 	gold_entity_idxs.append(torch.LongTensor(gold_entity_idx))
					new_ent_preds, pred_entity_idxs, new_entity_labels, gold_entity_idxs = self.eval(dev_corpus, dev_set)
					run, max_score_epoch, max_score, analysis = self.posteval(epoch, max_score_epoch, max_score, dev_corpus, new_ent_preds, pred_entity_idxs, new_entity_labels, gold_entity_idxs)
					if not run:
						jsondump(analysis, "runs/eld/%s/%s_best.json" % (self.model_name, self.model_name))
						break

		self.load_model()  # load best
		test_corpus = self.data.test_dataset
		test_set = self.initialize_dataset(test_corpus)
		new_ent_preds, pred_entity_idxs, new_entity_labels, gold_entity_idxs = self.eval(test_corpus, test_set)
		kb_expectation_score, total_score, in_kb_score, out_kb_score, no_surface_score, cluster_score, mapping_result_clustered, mapping_result_unclustered = self.evaluator.evaluate(test_corpus, torch.tensor(new_ent_preds).view(-1),
		                                                                                                                                                                              torch.tensor(pred_entity_idxs),
		                                                                                                                                                                              torch.cat(new_entity_labels, dim=-1).cpu(),
		                                                                                                                                                                              torch.cat(gold_entity_idxs, dim=-1).cpu())
		print()
		p, r, f = kb_expectation_score
		gl.logger.info("Test score")
		gl.logger.info("%s score: P %.2f, R %.2f, F1 %.2f" % ("KB Expectation", p * 100, r * 100, f * 100))
		for score_info, ((cp, cr, cf), (up, ur, uf)) in [["Total", total_score],
		                                                 ["in-KB", in_kb_score],
		                                                 ["out KB", out_kb_score],
		                                                 ["No surface", no_surface_score]]:
			gl.logger.info("%s score: Clustered - P %.2f, R %.2f, F1 %.2f, Unclustered - P %.2f, R %.2f, F1 %.2f" % (score_info, cp * 100, cr * 100, cf * 100, up * 100, ur * 100, uf * 100))
		gl.logger.info("Clustering score: %.2f" % (cluster_score * 100))
		analyze_data = self.data.analyze(test_corpus, torch.tensor(new_ent_preds).view(-1), torch.tensor(pred_entity_idxs), torch.cat(new_entity_labels).cpu(), torch.cat(gold_entity_idxs).cpu(),
		                                 (kb_expectation_score, total_score, in_kb_score, out_kb_score, no_surface_score, cluster_score, mapping_result_clustered, mapping_result_unclustered))
		jsondump(analyze_data, "runs/eld/%s/%s_test.json" % (self.model_name, self.model_name))

	def save_model(self):
		torch.save({
			"transformer": self.transformer.state_dict(),
			"binary"     : self.binary_classifier.state_dict(),
			"vector"     : self.vector_transformer.state_dict()
		}, self.model_path)

	def load_model(self):
		pretrained_weight = "bert-base-multilingual-cased"
		self.transformer = BertModel.from_pretrained(pretrained_weight).to(self.device)
		self.transformer.resize_token_embeddings(len(self.tokenizer))
		self.binary_classifier = FFNNEncoder(768, 1, 768, 2).to(self.device)
		self.vector_transformer = FFNNEncoder(768, 300, 768, 2).to(self.device)
		state_dict = torch.load(self.args.model_path)
		self.transformer.load_state_dict(state_dict["transformer"])
		self.binary_classifier.load_state_dict(state_dict["binary"])
		self.vector_transformer.load_state_dict(state_dict["vector"])

	def eval(self, corpus, batchs, batch_size=4):
		with torch.no_grad():
			self.transformer.eval()
			self.binary_classifier.eval()
			self.vector_transformer.eval()
			self.data.reset_new_entity()
			new_entity_labels = []
			gold_entity_idxs = []
			kb_scores = []
			preds = []
			for batch in split_to_batch(batchs, batch_size):
				encoded_sentence, in_kb_label, target_embedding, gold_entity_idx = zip(*batch)
				encoded_sequence = torch.stack(encoded_sentence).to(self.device)
				# attention_mask = torch.where(encoded_sequence > 0, torch.ones_like(encoded_sequence), torch.zeros_like(encoded_sequence))

				new_entity_label = torch.ByteTensor(in_kb_label).to(self.device)
				# print(new_entity_label.size())
				transformer_output = self.transformer(encoded_sequence)[0][:, 0, :]
				kb_score = self.binary_classifier(transformer_output)

				pred = self.vector_transformer(transformer_output)
				kb_scores.append(kb_score)
				preds.append(pred)

				new_entity_labels.append(new_entity_label)
				gold_entity_idxs.append(torch.LongTensor(gold_entity_idx))
			new_ent_preds, pred_entity_idxs = self.data.predict_entity(corpus.eld_items, kb_scores, preds, )
		return new_ent_preds, pred_entity_idxs, new_entity_labels, gold_entity_idxs

class ELDNoDiscovery(ELDSkeleton):
	def __init__(self, mode, args=None):
		super(ELDNoDiscovery, self).__init__(mode, "nodiscovery", train_new=True, train_args=args)
		# dev_batch = DataLoader(dataset=self.data.dev_dataset, batch_size=256, shuffle=False, num_workers=8)

		test_corpus = self.data.test_dataset
		new_ent_preds, pred_entity_idxs, new_entity_labels, gold_entity_idxs = self.eval(test_corpus)

		run, max_score_epoch, max_score, analysis = self.posteval(0, 0, 0, test_corpus.eld_items, new_ent_preds, pred_entity_idxs, new_entity_labels, gold_entity_idxs)

	def save_model(self):
		pass

	def load_model(self):
		pass

	def predict(self, data):
		pass

	def eval(self, corpus):
		new_entity_labels = [x.is_new_entity for x in corpus.eld_items]
		gold_entity_idxs = [x.entity_label_idx for x in corpus.eld_items]
		kb_scores = [0 for _ in corpus.eld_items]
		preds = [torch.zeros(1, 300).to(self.device, dtype=torch.float) for _ in corpus.eld_items]
		new_ent_preds, pred_entity_idxs = self.data.predict_entity(corpus.eld_items, torch.tensor(kb_scores), torch.cat(preds))
		return new_ent_preds, pred_entity_idxs, [torch.tensor(new_entity_labels)], [torch.tensor(gold_entity_idxs)]

class DictBasedELD(ELDSkeleton):
	"""
	개체 연결 -> 실패 -> 등록 -> 재연결 시도
	"""

	def __init__(self, mode, args):
		super(DictBasedELD, self).__init__(mode, "dictbased", train_args=args)

	def save_model(self):
		pass

	def load_model(self):
		pass

	def predict(self, *data):
		dataset = self.data.prepare(self.mode, *data, namu_only=True)
		for ent in dataset.eld_items:
			_, link_result = self.data.predict_entity([ent], output_as_idx=False, mark_nil=True)
			link_result = link_result[0]
			if link_result == "NOT_IN_CANDIDATE":
				self.data.surface_ent_dict.add_instance(ent.surface, ent.surface)
				link_result = ent.surface

			ent.eld_pred_entity = link_result
		if type(data[0]) is Corpus:
			return data[0]

		return [x.to_json() for x in dataset.eld_items]
