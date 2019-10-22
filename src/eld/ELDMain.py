import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .modules import SeparateEntityEncoder, JointEntityEncoder, VectorTransformer, FFNNEncoder, SelfAttentionEncoder
from .utils import ELDArgs, DataModule, Evaluator
from .. import GlobalValues as gl
from ..utils import jsondump, split_to_batch

import os
import random
from pytorch_transformers import BertTokenizer, BertModel

# noinspection PyMethodMayBeStatic
class ELDSkeleton:
	def __init__(self, mode: str, model_name: str, train_new=True, train_args: ELDArgs = None):
		assert mode in ["train", "eval", "demo"]
		self.model_name = model_name
		if mode != "train":
			self.args = ELDArgs(model_name)
		else:
			try:
				if train_new:
					if train_args is not None:
						self.args = train_args
					else:
						self.args = ELDArgs(model_name)
				else:
					self.args = ELDArgs.from_json("models/eld/%s_args.json" % model_name)
			except Exception:
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
		loss = None
		closs = F.binary_cross_entropy_with_logits(kb_score.squeeze(), new_entity_flag.float())
		if loss is not None:
			loss += closs
		else:
			loss = closs
		return loss

	def out_kb_loss(self, pred, new_entity_flag, label):
		loss = sum([F.mse_loss(p, g) if t == 0 else torch.zeros_like(F.mse_loss(p, g)) for t, p, g in zip(new_entity_flag, pred, label)])
		loss += sum([F.mse_loss(p, g) if t == 1 and torch.sum(g) != 0 else torch.zeros_like(F.mse_loss(p, g)) for t, p, g in zip(new_entity_flag, pred, label)])  # zero tensor는 일단 거름
		return loss

	def posteval(self, epoch, max_score, dev_corpus, new_ent_preds, pred_entity_idxs, new_entity_labels, gold_entity_idxs):
		kb_expectation_score, total_score, in_kb_score, out_kb_score, no_surface_score, cluster_score, mapping_result = self.evaluator.evaluate(dev_corpus, torch.tensor(new_ent_preds).view(-1), torch.tensor(pred_entity_idxs),
		                                                                                                                                        torch.cat(new_entity_labels).cpu(), torch.cat(gold_entity_idxs).cpu())
		print()
		for score_info, (p, r, f) in [["KB expectation", kb_expectation_score],
		                              ["Total", total_score],
		                              ["in-KB", in_kb_score],
		                              ["out KB", out_kb_score],
		                              ["No surface", no_surface_score]]:
			gl.logger.info("%s score: P %.2f, R %.2f, F1 %.2f" % (score_info, p * 100, r * 100, f * 100))
		gl.logger.info("Clustering score: %.2f" % (cluster_score * 100))
		if total_score[-1] > max_score:
			max_score = total_score[-1]
			max_score_epoch = epoch
			torch.save(self.transformer.state_dict(), self.model_path)
		gl.logger.info("Best epoch %d - Score %.2f" % (max_score_epoch, max_score * 100))
		if not os.path.isdir("runs/eld/%s" % self.model_name):
			os.mkdir("runs/eld/%s" % self.model_name)
		jsondump(self.data.analyze(dev_corpus, torch.tensor(new_ent_preds).view(-1), torch.tensor(pred_entity_idxs), torch.cat(new_entity_labels).cpu(), torch.cat(gold_entity_idxs).cpu(),
		                           (kb_expectation_score, total_score, in_kb_score, out_kb_score, no_surface_score, cluster_score, mapping_result)), "runs/eld/%s/%s_%d.json" % (self.model_name, self.model_name, epoch))
		if epoch - max_score_epoch > self.stop:
			gl.logger.info("No better performance for %d epoch - Training stop" % self.stop)
			return False
		return True

class ELD(ELDSkeleton):
	def __init__(self, mode: str, model_name: str, train_new=True, train_args: ELDArgs = None):
		super(ELD, self).__init__(mode, model_name, train_new, train_args)
		transformer_map = {"separate": SeparateEntityEncoder, "joint": JointEntityEncoder}
		args = self.args
		if mode != "train":
			self.transformer = transformer_map[args.transformer_mode] \
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

		self.transformer = transformer_map[args.transformer_mode] \
			(args.use_character_embedding, args.use_word_embedding, args.use_word_context_embedding, args.use_entity_context_embedding, args.use_relation_embedding, args.use_type_embedding,
			 args.character_encoder, args.word_encoder, args.word_context_encoder, args.entity_context_encoder, args.relation_encoder, args.type_encoder,
			 args.c_emb_dim, args.w_emb_dim, args.e_emb_dim, args.r_emb_dim, args.t_emb_dim,
			 args.c_enc_dim, args.w_enc_dim, args.wc_enc_dim, args.ec_enc_dim, args.r_enc_dim, args.t_enc_dim,
			 args.jamo_limit, args.word_limit, args.relation_limit).to(self.device)
		self.vector_transformer = VectorTransformer(self.transformer.max_input_dim, args.e_emb_dim, args.flags).to(self.device)

		jsondump(args.to_json(), "models/eld/%s_args.json" % model_name)
		gl.logger.info("ELD Model load complete")

	def train(self):
		gl.logger.info("Train start")
		batch_size = 256
		train_batch = DataLoader(dataset=self.data.train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
		dev_batch = DataLoader(dataset=self.data.dev_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
		dev_corpus = self.data.dev_corpus
		tqdmloop = tqdm(range(1, self.epochs + 1))
		discovery_optimizer = torch.optim.Adam(self.transformer.parameters(), lr=1e-3, weight_decay=1e-4)
		tensor_optimizer = torch.optim.Adam(self.vector_transformer.parameters(), lr=1e-3, weight_decay=1e-4)
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
				ce, cl, we, wl, lwe, lwl, rwe, rwl, lee, lel, ree, rel, re, rl, te, tl = [x.to(self.device, torch.float) if x is not None else None for x in batch[:-3]]
				new_entity_label, ee_label, gold_entity_idx = [x.to(self.device) for x in batch[-3:]]
				kb_score, pred, attn_mask = self.transformer(ce, cl, we, wl, lwe, lwl, rwe, rwl, lee, lel, ree, rel, re, rl, te, tl)
				# pred = torch.tensor(pred, requires_grad=False).detach()
				discovery_loss_val = self.discovery_loss(kb_score, new_entity_label)
				discovery_loss_val.backward()
				discovery_optimizer.step()
				pred = self.vector_transformer(pred.detach())
				tensor_loss_val = self.out_kb_loss(pred, new_entity_label, ee_label)
				tensor_loss_val.backward()
				tensor_optimizer.step()
				tqdmloop.set_description("Epoch %d: Loss %.4f" % (epoch, discovery_loss_val + tensor_loss_val))
				ne.append(new_entity_label)
				ei.append(gold_entity_idx)
				pr.append(pred)
			self.data.update_new_entity_embedding(torch.cat(ne), torch.cat(ei), torch.cat(pr))

			if epoch % self.eval_per_epoch == 0:
				self.transformer.eval()
				self.vector_transformer.eval()
				self.data.reset_new_entity()
				new_ent_preds = []
				pred_entity_idxs = []
				new_entity_labels = []
				gold_entity_idxs = []
				dev_batch_start_idx = 0
				for batch in dev_batch:
					ce, cl, we, wl, lwe, lwl, rwe, rwl, lee, lel, ree, rel, re, rl, te, tl = [x.to(self.device, torch.float32) if x is not None else None for x in batch[:-3]]
					kb_score, pred, attn_mask = self.transformer(ce, cl, we, wl, lwe, lwl, rwe, rwl, lee, lel, ree, rel, re, rl, te, tl)
					kb_score = torch.sigmoid(kb_score)
					pred = self.vector_transformer(pred)
					new_entity_label, _, gold_entity_idx = [x.to(self.device) for x in batch[-3:]]
					new_ent_pred, entity_idx = self.data.predict_entity(kb_score, pred, dev_corpus.eld_get_item(slice(dev_batch_start_idx, dev_batch_start_idx + batch_size)))
					new_ent_preds += new_ent_pred
					pred_entity_idxs += entity_idx
					new_entity_labels.append(new_entity_label)
					gold_entity_idxs.append(gold_entity_idx)
				if not self.posteval(epoch, max_score, dev_corpus, new_ent_preds, pred_entity_idxs, new_entity_labels, gold_entity_idxs): break
				# kb_expectation_score, total_score, in_kb_score, out_kb_score, no_surface_score, cluster_score, mapping_result = self.evaluator.evaluate(dev_corpus, torch.tensor(new_ent_preds).view(-1), torch.tensor(pred_entity_idxs),
				#                                                                                                                                         torch.cat(new_entity_labels).cpu(), torch.cat(gold_entity_idxs).cpu())
				# print()
				# for score_info, (p, r, f) in [["KB expectation", kb_expectation_score],
				#                               ["Total", total_score],
				#                               ["in-KB", in_kb_score],
				#                               ["out KB", out_kb_score],
				#                               ["No surface", no_surface_score]]:
				# 	gl.logger.info("%s score: P %.2f, R %.2f, F1 %.2f" % (score_info, p * 100, r * 100, f * 100))
				# gl.logger.info("Clustering score: %.2f" % (cluster_score * 100))
				# if total_score[-1] > max_score:
				# 	max_score = total_score[-1]
				# 	max_score_epoch = epoch
				# 	torch.save(self.transformer.state_dict(), self.model_path)
				# gl.logger.info("Best epoch %d - Score %.2f" % (max_score_epoch, max_score * 100))
				# if not os.path.isdir("runs/eld/%s" % self.model_name):
				# 	os.mkdir("runs/eld/%s" % self.model_name)
				# jsondump(self.data.analyze(dev_corpus, torch.tensor(new_ent_preds).view(-1), torch.tensor(pred_entity_idxs), torch.cat(new_entity_labels).cpu(), torch.cat(gold_entity_idxs).cpu(),
				#                            (kb_expectation_score, total_score, in_kb_score, out_kb_score, no_surface_score, cluster_score, mapping_result)), "runs/eld/%s/%s_%d.json" % (self.model_name, self.model_name, epoch))
				# if epoch - max_score_epoch > self.stop:
				# 	gl.logger.info("No better performance for %d epoch - Training stop" % self.stop)
				# 	break
	def predict(self, data, register=True):
		self.transformer.eval()
		pred_embedding = self.transformer(data)
		pred_embedding.repeat(len(self.entity_index))
		cos_sim = F.cosine_similarity(pred_embedding, self.entity_embedding)
		if cos_sim.max() > self.map_threshold:
			target_ind = cos_sim.argmax(dim=-1).cpu().data
			target = self.i2e[target_ind]
			if self.modify_entity_embedding:
				pred_embedding *= self.modify_entity_embedding_weight
				self.entity_embedding += torch.stack([torch.zeros(self.entity_embedding_dim) for _ in range(target_ind - 1)] + [pred_embedding] + [torch.zeros(self.entity_embedding_dim) for _ in range(len(self.i2e) - target_ind - 1)])
			return target
		if register:
			register_form = self.register_new_entity(data, pred_embedding)
			return register_form
		return None

	def register_new_entity(self, surface, entity_embedding):
		idx = len(self.entity_index)
		register_form = "__" + surface.replace(" ", "_")
		self.entity_index[register_form] = idx
		self.i2e[idx] = register_form
		self.entity_embedding = torch.cat((self.entity_embedding, entity_embedding.unsqueeze(0)), 0)
		return register_form

	def __call__(self, data):
		return self.predict(data)



# noinspection PyMethodMayBeStatic
class BertBasedELD(ELDSkeleton):
	def __init__(self, mode: str, model_name: str, train_new=True, train_args: ELDArgs = None):
		super(BertBasedELD, self).__init__(mode, model_name, train_new, train_args)
		pretrained_weight = "bert-base-multilingual-cased"
		self.tokenizer = BertTokenizer.from_pretrained(pretrained_weight).to(self.device)
		self.tokenizer.add_special_tokens({"cls_token": "<CLS>", "additional_special_tokens": ["<e>", "</e>"], "sep_token": "<SEP>"})
		self.transformer = BertTokenizer.from_pretrained(pretrained_weight).to(self.device)
		self.binary_classifier = FFNNEncoder(512, 1, 512, 2)
		self.vector_transformer = FFNNEncoder(512, 300, 512, 2)

	def initialize_dataset(self, corpus):
		s = []
		for item in corpus.eld_items:
			parent_sentence = item.parent_sentence
			marked_sentence = "<CLS>" + parent_sentence.original_sentence[:item.char_ind] + "<e>" + item.surface + "</e>" + parent_sentence.original_sentence[item.char_ind + len(item.surface):] + "<SEP>"
			new_entity_label = item.is_new_entity
			target_embedding = item.entity_label_embedding
			gold_entity_idx = item.entity_label_idx
			s.append((marked_sentence, new_entity_label, target_embedding, gold_entity_idx))
		return s

	def train(self):
		gl.logger.info("Train start")
		batch_size = 32
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
			for batch in split_to_batch(train_set, batch_size):
				marked_sentence, in_kb_label, target_embedding, _ = zip(*batch)
				transformer_input, attention_mask = torch.stack([self.tokenizer.encode_plus(x, max_length=512, return_tensors="pt") for x in marked_sentence]).to(self.device)
				new_entity_label = torch.stack(in_kb_label).to(self.device)
				target_embedding = torch.stack(target_embedding).to(self.device)
				transformer_output = self.transformer(transformer_input, attention_mask)[0][:, 0, :]
				kb_score = self.binary_classifier(transformer_output)
				discovery_loss_val = self.discovery_loss(kb_score, new_entity_label)
				discovery_loss_val.backward()

				pred = self.vector_transformer(transformer_output)
				tensor_loss_val = self.out_kb_loss(pred, new_entity_label, target_embedding)
				tensor_loss_val.backward()
				optimizer.step()
			if epoch % self.eval_per_epoch == 0:
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
					marked_sentence, in_kb_label, target_embedding, gold_entity_idx = zip(*batch)
					transformer_input, attention_mask = torch.stack([self.tokenizer.encode_plus(x, max_length=512, return_tensors="pt") for x in marked_sentence]).to(self.device)
					new_entity_label = torch.stack(in_kb_label).to(self.device)
					target_embedding = torch.stack(target_embedding).to(self.device)
					transformer_output = self.transformer(transformer_input, attention_mask)[0][:, 0, :]
					kb_score = self.binary_classifier(transformer_output)
					pred = self.vector_transformer(target_embedding)
					new_ent_pred, entity_idx = self.data.predict_entity(kb_score, pred, dev_corpus.eld_get_item(slice(dev_batch_start_idx, dev_batch_start_idx + batch_size)))
					new_ent_preds += new_ent_pred
					pred_entity_idxs += entity_idx
					new_entity_labels.append(new_entity_label)
					gold_entity_idxs.append(gold_entity_idx)
				if not self.posteval(epoch, max_score, dev_corpus, new_ent_preds, pred_entity_idxs, new_entity_labels, gold_entity_idxs): break

