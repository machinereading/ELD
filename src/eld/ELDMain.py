import os
import random
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

from .modules import SeparateEntityEncoder, VectorTransformer, FFNNEncoder, VectorTransformer2, VectorTransformer3
from .utils import ELDArgs, DataModule, Evaluator
from .. import GlobalValues as gl
from ..utils import jsondump, split_to_batch
import traceback
# from tabulate import tabulate

# noinspection PyMethodMayBeStatic
class ELDSkeleton(ABC):
	def __init__(self, mode: str, model_name: str, train_new=True, train_args: ELDArgs = None):
		assert mode in ["train", "eval", "demo"]
		self.model_name = model_name
		if mode != "train":
			self.args = ELDArgs.from_json("models/eld/%s_args.json" % model_name)
			self.load_model()
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
		self.device = self.args.device = "cuda" if torch.cuda.is_available() and mode != "demo" else "cpu"

		self.args.mode = mode
		self.data = DataModule(mode, self.args)
		self.evaluator = Evaluator(self.args, self.data)
		self.stop = self.args.early_stop

	def discovery_loss(self, kb_score, new_entity_flag):
		return F.binary_cross_entropy_with_logits(kb_score.view(-1), new_entity_flag.float())

	def out_kb_loss(self, pred, new_entity_flag, label):
		l1 = sum([F.mse_loss(p, g) if t == 0 else torch.zeros_like(F.mse_loss(p, g)) for t, p, g in zip(new_entity_flag, pred, label)])
		l2 = sum([F.mse_loss(p, g) if t == 1 and torch.sum(g) != 0 else torch.zeros_like(F.mse_loss(p, g)) for t, p, g in zip(new_entity_flag, pred, label)])  # zero tensor는 일단 거름
		# return sum([F.mse_loss(p, g) for t, p, g in zip(new_entity_flag, pred, label) if torch.sum(g) != 0])
		return l1 + l2

	def posteval(self, epoch, max_score_epoch, max_score, dev_corpus, new_ent_preds, pred_entity_idxs, new_entity_labels, gold_entity_idxs):
		kb_expectation_score, total_score, in_kb_score, out_kb_score, no_surface_score, cluster_score, mapping_result_clustered, mapping_result_unclustered = self.evaluator.evaluate(dev_corpus, torch.tensor(new_ent_preds).view(-1), torch.tensor(pred_entity_idxs),
		                                                                                                                                        torch.cat(new_entity_labels, dim=-1).cpu(), torch.cat(gold_entity_idxs, dim=-1).cpu())
		print()
		p, r, f = kb_expectation_score
		gl.logger.info("%s score: P %.2f, R %.2f, F1 %.2f" % ("KB Expectation", p * 100, r * 100, f * 100))
		for score_info, ((cp, cr, cf), (up, ur, uf)) in [["Total", total_score],
		                              ["in-KB", in_kb_score],
		                              ["out KB", out_kb_score],
		                              ["No surface", no_surface_score]]:
			gl.logger.info("%s score: Clustered - P %.2f, R %.2f, F1 %.2f, Unclustered - P %.2f, R %.2f, F1 %.2f" % (score_info, cp * 100, cr * 100, cf * 100, up * 100, ur * 100, uf * 100))
		gl.logger.info("Clustering score: %.2f" % (cluster_score * 100))
		if kb_expectation_score[-1] > max_score:
			max_score = kb_expectation_score[-1]
			max_score_epoch = epoch
			self.save_model()
		gl.logger.info("Best epoch %d - Score %.2f" % (max_score_epoch, max_score * 100))
		if not os.path.isdir("runs/eld/%s" % self.model_name):
			os.mkdir("runs/eld/%s" % self.model_name)
		analyze_data = self.data.analyze(dev_corpus, torch.tensor(new_ent_preds).view(-1), torch.tensor(pred_entity_idxs), torch.cat(new_entity_labels).cpu(), torch.cat(gold_entity_idxs).cpu(),
		                           (kb_expectation_score, total_score, in_kb_score, out_kb_score, no_surface_score, cluster_score, mapping_result_clustered, mapping_result_unclustered))
		jsondump(analyze_data, "runs/eld/%s/%s_%d.json" % (self.model_name, self.model_name, epoch))
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
	def predict(self, data):
		pass

	def __call__(self, data):
		return self.predict(data)


class ELD(ELDSkeleton):
	def __init__(self, mode: str, model_name: str, train_new=True, train_args: ELDArgs = None):
		super(ELD, self).__init__(mode, model_name, train_new, train_args)
		args = self.args
		if mode != "train":
			self.transformer = SeparateEntityEncoder \
				(args.use_character_embedding, args.use_word_embedding, args.use_word_context_embedding, args.use_entity_context_embedding, args.use_relation_embedding, args.use_type_embedding,
				 args.character_encoder, args.word_encoder, args.word_context_encoder, args.entity_context_encoder, args.relation_encoder, args.type_encoder,
				 args.c_emb_dim, args.w_emb_dim, args.e_emb_dim, args.r_emb_dim, args.t_emb_dim,
				 args.c_enc_dim, args.w_enc_dim, args.wc_enc_dim, args.ec_enc_dim, args.r_enc_dim, args.t_enc_dim,
				 args.jamo_limit, args.word_limit, args.relation_limit).to(self.device)
			try:
				self.transformer.load_state_dict(torch.load(args.model_path))
			except:
				gl.logger.critical("No model exists!")
			self.modify_entity_embedding = args.modify_entity_embedding
			self.modify_entity_embedding_weight = args.modify_entity_embedding_weight
		self.entity_index = {}
		self.i2e = {v: k for k, v in self.entity_index.items()}
		self.use_explicit_kb_classifier = args.use_explicit_kb_classifier
		self.train_embedding = args.train_embedding
		if mode == "test":
			self.entity_embedding = self.data.entity_embedding.weight
			self.entity_embedding_dim = self.entity_embedding.size()[-1]
		self.map_threshold = args.out_kb_threshold

		self.transformer = SeparateEntityEncoder \
			(args.use_character_embedding, args.use_word_embedding, args.use_word_context_embedding, args.use_entity_context_embedding, args.use_relation_embedding, args.use_type_embedding,
			 args.character_encoder, args.word_encoder, args.word_context_encoder, args.entity_context_encoder, args.relation_encoder, args.type_encoder,
			 args.c_emb_dim, args.w_emb_dim, args.e_emb_dim, args.r_emb_dim, args.t_emb_dim,
			 args.c_enc_dim, args.w_enc_dim, args.wc_enc_dim, args.ec_enc_dim, args.r_enc_dim, args.t_enc_dim,
			 args.jamo_limit, args.word_limit, args.relation_limit).to(self.device)
		self.vector_transformer = VectorTransformer3(self.transformer.max_input_dim, args.e_emb_dim, args.flags).to(self.device)

		jsondump(args.to_json(), "models/eld/%s_args.json" % model_name)
		gl.logger.info("ELD Model load complete")

	def train(self):
		gl.logger.info("Train start")
		batch_size = 256
		train_batch = DataLoader(dataset=self.data.train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
		dev_batch = DataLoader(dataset=self.data.dev_dataset, batch_size=4, shuffle=False, num_workers=8)
		dev_corpus = self.data.dev_dataset
		tqdmloop = tqdm(range(1, self.epochs + 1))
		discovery_optimizer = torch.optim.SGD(self.transformer.parameters(), lr=1e-3, weight_decay=1e-4)
		tensor_optimizer = torch.optim.Adam(self.vector_transformer.parameters(), lr=1e-3)
		# params = list(self.vector_transformer.parameters()) + list(self.transformer.parameters())
		# optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)
		# optimizer = torch.optim.SGD(params, lr=0.001, weight_decay=1e-4)
		max_score = 0
		max_score_epoch = 0
		for epoch in tqdmloop:
			self.transformer.train()
			self.vector_transformer.train()
			ne = []
			ei = []
			pr = []
			for batch in train_batch:
				discovery_optimizer.zero_grad()
				tensor_optimizer.zero_grad()
				# optimizer.zero_grad()
				ce, cl, we, wl, lwe, lwl, rwe, rwl, lee, lel, ree, rel, re, rl, te, tl = [x.to(self.device, torch.float) if x is not None else None for x in batch[:-4]]
				new_entity_label, ee_label, gold_entity_idx = [x.to(self.device) for x in batch[-4:-1]]
				kb_score, pred = self.transformer(ce, cl, we, wl, lwe, lwl, rwe, rwl, lee, lel, ree, rel, re, rl, te, tl)
				discovery_loss_val = self.discovery_loss(kb_score, new_entity_label)
				discovery_loss_val.backward()
				discovery_optimizer.step()
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

			if epoch % self.eval_per_epoch == 0:
				self.transformer.eval()
				self.vector_transformer.eval()
				self.data.reset_new_entity() # registered entity를 전부 지우고 다시 처음부터 등록 시퀀스 시작
				with torch.no_grad():
					# new_ent_preds = []
					# pred_entity_idxs = []
					new_entity_labels = []
					gold_entity_idxs = []
					kb_scores = []
					preds = []
					dev_idxs = []
					dev_batch_start_idx = 0
					for batch in dev_batch:
						ce, cl, we, wl, lwe, lwl, rwe, rwl, lee, lel, ree, rel, re, rl, te, tl = [x.to(self.device, torch.float32) if x is not None else None for x in batch[:-4]]
						kb_score, pred = self.transformer(ce, cl, we, wl, lwe, lwl, rwe, rwl, lee, lel, ree, rel, re, rl, te, tl)
						kb_score = torch.sigmoid(kb_score)
						# print(pred)
						pred = self.vector_transformer(pred.detach(), eval=False) # TODO why all same tensor?
						# print(pred)
						new_entity_label, _, gold_entity_idx = [x.to(self.device) for x in batch[-4:-1]]
						dev_idxs.append(batch[-1])

						kb_scores.append(kb_score)
						preds.append(pred)
						new_entity_labels.append(new_entity_label)
						gold_entity_idxs.append(gold_entity_idx)
						dev_batch_start_idx += batch_size
					for l_s, i_s in zip(gold_entity_idxs, dev_idxs):
						for l, i in zip(l_s, i_s):
							v = dev_corpus.eld_items[i]
							assert v.entity_label_idx == l
						assert v.entity_label_idx == l, "%d/%d/%s" % (l, v.entity_label_idx, v.entity)
					new_ent_preds, pred_entity_idxs = self.data.predict_entity(torch.cat(kb_scores), torch.cat(preds), dev_corpus.eld_items)
					run, max_score_epoch, max_score, analysis = self.posteval(epoch, max_score_epoch, max_score, dev_corpus.eld_items, new_ent_preds, pred_entity_idxs, new_entity_labels, gold_entity_idxs)
					if not run:
						jsondump(analysis, "runs/eld/%s/%s_best.json" % (self.model_name, self.model_name))
						break

	# def register_new_entity(self, surface, entity_embedding):
	# 	idx = len(self.entity_index)
	# 	register_form = "__" + surface.replace(" ", "_")
	# 	self.entity_index[register_form] = idx
	# 	self.i2e[idx] = register_form
	# 	self.entity_embedding = torch.cat((self.entity_embedding, entity_embedding.unsqueeze(0)), 0)
	# 	return register_form

	def save_model(self):
		torch.save(self.transformer.state_dict(), self.model_path)

	def load_model(self):
		args = self.args
		self.transformer = SeparateEntityEncoder \
			(args.use_character_embedding, args.use_word_embedding, args.use_word_context_embedding, args.use_entity_context_embedding, args.use_relation_embedding, args.use_type_embedding,
			 args.character_encoder, args.word_encoder, args.word_context_encoder, args.entity_context_encoder, args.relation_encoder, args.type_encoder,
			 args.c_emb_dim, args.w_emb_dim, args.e_emb_dim, args.r_emb_dim, args.t_emb_dim,
			 args.c_enc_dim, args.w_enc_dim, args.wc_enc_dim, args.ec_enc_dim, args.r_enc_dim, args.t_enc_dim,
			 args.jamo_limit, args.word_limit, args.relation_limit).to(self.device)
		try:
			self.transformer.load_state_dict(torch.load(args.model_path))
		except:
			gl.logger.critical("No model exists!")

	def predict(self, data):
		self.transformer.eval()
		self.vector_transformer.eval()
		data = self.data.prepare(data)

		with torch.no_grad():
			kb_scores = []
			preds = []
			for batch in split_to_batch(data, 512):
				batch = [x.tensor for x in batch]
				ce, cl, we, wl, lwe, lwl, rwe, rwl, lee, lel, ree, rel, re, rl, te, tl = [x.to(self.device, torch.float32) if x is not None else None for x in batch[:-3]]
				kb_score, pred, attn_mask = self.transformer(ce, cl, we, wl, lwe, lwl, rwe, rwl, lee, lel, ree, rel, re, rl, te, tl)
				kb_score = torch.sigmoid(kb_score)
				pred = self.vector_transformer(pred)
				kb_scores.append(kb_score)
				preds.append(pred)
			new_ent_preds, pred_entity_idxs = self.data.predict_entity(torch.cat(kb_scores), torch.cat(preds), data)

# noinspection PyMethodMayBeStatic
class BertBasedELD(ELDSkeleton):
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
			marked_sentence = "[CLS] " + parent_sentence.original_sentence[:item.char_ind] + " [e] " + item.surface + " [/e] " + parent_sentence.original_sentence[item.char_ind + len(item.surface):] + " [SEP]"
			new_entity_label = item.is_new_entity
			target_embedding = item.entity_label_embedding
			gold_entity_idx = item.entity_label_idx
			encoded_sentence = pad(self.tokenizer.encode(marked_sentence, max_length=512, return_tensors="pt", add_special_tokens=True).squeeze(0), 512)
			s.append((encoded_sentence, new_entity_label, target_embedding, gold_entity_idx))
		return s

	def train(self):
		gl.logger.info("Train start")
		batch_size = 4
		train_corpus = self.data.train_corpus
		dev_corpus = self.data.dev_corpus
		tqdmloop = tqdm(range(1, self.epochs + 1))
		optimizer = torch.optim.Adam(self.transformer.parameters(), lr=1e-3, weight_decay=1e-4)
		max_score = 0
		max_score_epoch = 0
		# initialize train and dev set
		train_set = self.initialize_dataset(train_corpus)
		dev_set = self.initialize_dataset(dev_corpus)

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
				attention_mask = torch.where(encoded_sequence > 0, torch.ones_like(encoded_sequence), torch.zeros_like(encoded_sequence))
				# transformer_input = torch.stack(encoded_sequence).to(self.device)

				new_entity_label = torch.ByteTensor(in_kb_label).to(self.device)
				target_embedding = torch.stack(target_embedding).to(self.device)
				transformer_output = self.transformer(encoded_sequence, attention_mask)[0][:, 0, :]
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
					new_ent_preds = []
					pred_entity_idxs = []
					new_entity_labels = []
					gold_entity_idxs = []
					dev_batch_start_idx = 0
					for batch in split_to_batch(dev_set, batch_size):
						encoded_sentence, in_kb_label, target_embedding, gold_entity_idx = zip(*batch)
						encoded_sequence = torch.stack(encoded_sentence).to(self.device)
						attention_mask = torch.where(encoded_sequence > 0, torch.ones_like(encoded_sequence), torch.zeros_like(encoded_sequence))

						new_entity_label = torch.ByteTensor(in_kb_label).to(self.device)
						# print(new_entity_label.size())
						transformer_output = self.transformer(encoded_sequence, attention_mask)[0][:, 0, :]
						kb_score = self.binary_classifier(transformer_output)

						pred = self.vector_transformer(transformer_output)
						new_ent_pred, entity_idx = self.data.predict_entity(kb_score, pred, dev_corpus.eld_get_item(slice(dev_batch_start_idx, dev_batch_start_idx + batch_size)))
						new_ent_preds += new_ent_pred
						pred_entity_idxs += entity_idx
						new_entity_labels.append(new_entity_label)
						gold_entity_idxs.append(torch.LongTensor(gold_entity_idx))
					run, max_score_epoch, max_score, analysis = self.posteval(epoch, max_score_epoch, max_score, dev_corpus, new_ent_preds, pred_entity_idxs, new_entity_labels, gold_entity_idxs)
					if not run:
						jsondump(analysis, "runs/eld/%s/%s_best.json" % (self.model_name, self.model_name))
						break
		self.test()

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


class ELDNoDiscovery(ELDSkeleton):

	def save_model(self):
		pass

	def load_model(self):
		pass

	def predict(self, data):
		pass

