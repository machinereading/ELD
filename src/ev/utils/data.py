import os
import random
import traceback

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .Tokenizer import Tokenizer
from ... import GlobalValues as gl
from ...ds import *
from ...utils import readfile, jsonload, jsondump, TimeUtil, split_to_batch

class SentenceGenerator(Dataset):
	def __init__(self, corpus):
		self.corpus = corpus

	def __len__(self):
		return sum(self.corpus.tagged_voca_lens)

	def __getitem__(self, ind):
		voca = self.corpus[ind]
		return voca.lctxw_ind, voca.rctxw_ind, voca.lctxe_ind, voca.rctxe_ind, voca.error_type + 1

class ClusterGenerator(Dataset):
	def __init__(self, corpus, for_train=False, filter_nik=False):
		assert not (for_train and filter_nik)
		self.corpus = corpus

		if for_train:
			# add more labels if label is biased
			l1 = [x for x in corpus.cluster_list if type(x) is Cluster]
			l0 = [x for x in corpus.cluster_list if type(x) is FakeCluster]
			l0_l1_ratio = round(len(l0) / len(l1))
			l1_l0_ratio = round(len(l1) / len(l0))
			for _ in range(l0_l1_ratio - 1):
				self.corpus.additional_cluster += l1
			for _ in range(l1_l0_ratio - 1):
				self.corpus.additional_cluster += l0
		if filter_nik:
			self.corpus.cluster = {k: v for k, v in corpus.cluster.items() if not v.is_in_kb}

	def __len__(self):
		return len(self.corpus.cluster_list)

	def __getitem__(self, ind):
		cluster = self.corpus.get_cluster_by_index(ind)
		return cluster.vocab_tensors

class ClusterContextSetGenerator(Dataset):
	def __init__(self, corpus):
		self.corpus = corpus

	def __len__(self):
		return len(self.corpus.cluster)

	def __getitem__(self, item):
		pass

class DataModule:
	def __init__(self, mode, args):
		gl.logger.info("Initializing EV DataModule")
		self.mode = mode
		# 0 for oov, 1 for out of range
		self.make_word_tensor = args.word_embedding_type == "glove"
		self.make_entity_tensor = args.entity_embedding_type == "glove"

		w2i = {w: i for i, w in
		       enumerate(readfile(args.word_embedding_path + ".word"))} if self.make_word_tensor else None
		self.wt = Tokenizer(args.word_embedding_type, w2i)
		self.wt_pad = len(w2i) + 1

		e2i = {e: i for i, e in
		       enumerate(readfile(args.entity_embedding_path + ".word"))} if self.make_entity_tensor else None
		self.et = Tokenizer(args.entity_embedding_type, e2i)
		self.et_pad = len(e2i) + 1
		self.chunk_size = args.chunk_size

		self.kb = [x for x in readfile(args.kb)]
		# check if we can load data from pre-defined cluster
		self.batch_size = args.batch_size
		self.ctx_window_size = args.ctx_window_size
		self.filter_data_tokens = args.filter_data_tokens
		self.target_device = args.device
		self.fake_er_rate = args.fake_er_rate
		self.fake_el_rate = args.fake_el_rate
		self.fake_ec_rate = args.fake_ec_rate
		self.fc_ratio = args.fake_cluster_rate
		self.corpus = None
		if mode == "train":
			if args.data_load_path is not None:
				self.data_load_path = args.data_load_path
				corpus_path = args.data_load_path
				try:
					path = "/".join(corpus_path.split("/")[:-1]) + "/"
					file_prefix = corpus_path.split("/")[-1]
					sentence = []
					cluster = []
					if len(os.listdir(path)) == 0:
						raise FileNotFoundError("No save file in dir %s" % corpus_path)
					for item in os.listdir(path):
						if not item.startswith(file_prefix): continue
						if "sentence" in item:
							sentence += jsonload(path + item)
					self.corpus = Corpus.from_json(sentence)
				except FileNotFoundError:
					traceback.print_exc()
				except:
					traceback.print_exc()
					import sys
					sys.exit(1)

			else:
				self.data_path = args.data_path
				self.generate_data()
			self.mark_kb(self.corpus)
			self.generate_fake_cluster(self.corpus)
			self.generate_cluster_vocab_tensors(self.corpus,
			                                    max_voca_restriction=getattr(args, "max_voca_restriction", -1),
			                                    max_jamo_restriction=getattr(args, "max_jamo_restriction", -1))

	@TimeUtil.measure_time
	def generate_data(self):
		# generate ev dataset from given corpus
		# corpus: path to corpus

		# pre-train distribution
		# generate clusters

		self.corpus = Corpus.load_corpus(self.data_path)

		# extract fake tokens
		self.fake_tokens = []
		for sentence in self.corpus:
			self.extract_fake_tokens(sentence)
		# distribute fake tokens into training cluster
		free_token_slot = 0
		for cluster in self.corpus.cluster.values():
			if len(cluster) < 2: continue
			if len(self.fake_tokens) == 0: break
			free_token_slot += len(cluster) // 2
			for i in range(len(cluster) // 2):
				if len(self.fake_tokens) == 0: break
				cluster.add_elem(self.fake_tokens.pop(0))
		if self.data_load_path is not None:
			self.save(self.data_load_path)

	def extract_fake_tokens(self, sentence):
		for token in sentence:
			if not token.is_entity and len(token) >= 2 and random.random() < self.fake_er_rate:
				token.error_type = 0
				self.fake_tokens.append(token)
			if token.is_entity and token.entity_in_kb and random.random() < self.fake_el_rate:
				token.error_type = 1
				self.fake_tokens.append(token)

	@TimeUtil.measure_time
	def generate_fake_cluster(self, corpus):
		# call after corpus and cluster is generated
		assert corpus is not None
		gl.logger.info("Generating fake clusters")
		dark_entity_len = len(list(filter(lambda x: not x.is_in_kb, corpus.cluster_list)))
		added_cluster_count = 0
		# generate fake cluster by adding same surface form entities
		suf_tok_dict = {}
		suf_tok_dict_nae = {}
		naes = {}
		for sentence in corpus:
			for token in sentence:
				if token.surface == "": continue
				if token.is_entity:
					if token.surface not in suf_tok_dict:
						suf_tok_dict[token.surface] = []
					suf_tok_dict[token.surface].append(token)
				else:
					if token.surface not in suf_tok_dict_nae:
						suf_tok_dict_nae[token.surface] = []
					suf_tok_dict_nae[token.surface].append(token)
				if token.entity == "NOT_AN_ENTITY":
					if token.surface not in naes:
						naes[token.surface] = []
					naes[token.surface].append(token)
		for surface, tokens in naes.items():
			if surface == "": continue
			newc = FakeCluster(surface + "_fake")
			for token in tokens[:100]:
				newc.add_elem(token)
			corpus.additional_cluster.append(newc)
			added_cluster_count += 1
			if added_cluster_count > len(corpus.cluster) * self.fc_ratio: break
		for surface, tokens in suf_tok_dict.items():
			entity_token_dict = {}
			for token in tokens[:100]:
				if token.entity not in entity_token_dict:
					entity_token_dict[token.entity] = []
				entity_token_dict[token.entity].append(token)
			if len(entity_token_dict) < 2: continue  # single cluster -> not a fake cluster
			lens = [len(x) for x in entity_token_dict.values()]
			s = sum(lens)
			m = max(lens)
			if s - m < m // 3: continue  # not enough invalid values to make fake cluster
			newc = FakeCluster(surface+"_fake")
			for token in tokens[:100]:
				newc.add_elem(token)
			corpus.additional_cluster.append(newc)
			added_cluster_count += 1
			if added_cluster_count > len(corpus.cluster) * self.fc_ratio: break

		# generate fake cluster with not_an_entity items
		for surface, tokens in suf_tok_dict_nae.items():
			if len(tokens) < 10: continue
			newc = FakeCluster(surface+"_fake")
			for token in tokens[:100]:
				newc.add_elem(token)
			corpus.additional_cluster.append(newc)
			added_cluster_count += 1
			if added_cluster_count > len(corpus.cluster) * self.fc_ratio: break
		return corpus

	def save(self, path):
		j = self.corpus.to_json()

		for i, item in enumerate(split_to_batch(j, 1000)):
			jsondump(item, path + "_sentence_%d.json" % i)

	@classmethod
	def from_predefined_cluster(cls, corpus_path):
		gen = cls(None, init=False)
		path = "/".join(corpus_path.split("/")[:-1]) + "/"
		file_prefix = corpus_path.split("/")[-1]
		sentence = []
		cluster = []
		for item in os.listdir(path):
			if not item.startswith(file_prefix): continue
			if "sentence" in item:
				sentence += jsonload(path + item)
			elif "cluster" in item:
				cluster += jsonload(path + item)
		gen.corpus = Corpus.from_json({"sentence": sentence, "cluster": cluster})

		return gen

	def mark_kb(self, corpus):
		gl.logger.info("Filtering KB Clusters")
		for cluster in corpus.cluster_list:
			cluster.is_in_kb = cluster.target_entity in self.kb
		corpus.cluster = {k: v for k, v in corpus.cluster.items() if not v.is_in_kb}
		gl.logger.debug("Remaining clusters: %d" % len(corpus.cluster_list))

	@TimeUtil.measure_time
	def generate_cluster_vocab_tensors(self, corpus, max_voca_restriction=None, max_jamo_restriction=None):
		gl.logger.info("Generating Vocab tensors...")
		invalid_cluster = []
		for cluster in tqdm(corpus.cluster_list, desc="Generating vocab tensors"):
			for vocab in cluster:
				if vocab.tagged: continue
				lctxw_ind = self.wt(vocab.lctx[-10:])[-self.ctx_window_size:]
				vocab.lctxw_ind = [self.wt_pad for _ in range(self.ctx_window_size - len(lctxw_ind))] + lctxw_ind

				rctxw_ind = self.wt(vocab.rctx[:10])[:self.ctx_window_size]
				vocab.rctxw_ind = ([self.wt_pad for _ in range(self.ctx_window_size - len(rctxw_ind))] + rctxw_ind)[::-1]

				lctxe_ind = self.et(vocab.lctx_ent[-10:])[-self.ctx_window_size:]
				vocab.lctxe_ind = [self.et_pad for _ in range(self.ctx_window_size - len(lctxe_ind))] + lctxe_ind

				rctxe_ind = self.et(vocab.rctx_ent[:10])[:self.ctx_window_size]
				vocab.rctxe_ind = ([self.et_pad for _ in range(self.ctx_window_size - len(rctxe_ind))] + rctxe_ind)[::-1]
				vocab.tagged = True

		# add padding
		gl.logger.info("Add padding...")
		max_voca = max([len(x) for x in corpus.cluster_list])
		if max_voca_restriction is not None and max_voca_restriction > 0:
			max_voca = max_voca_restriction
		max_voca += self.chunk_size - max_voca % self.chunk_size if max_voca % self.chunk_size > 0 else 0

		max_jamo = max([x.max_jamo for x in corpus.cluster_list])
		if max_jamo_restriction is not None and max_jamo_restriction > 0:
			max_jamo = max_jamo_restriction

		gl.logger.debug("Max vocabulary in cluster(with padding): %d" % max_voca)
		gl.logger.debug("Max jamo in word: %d" % max_jamo)
		for cluster in tqdm(corpus.cluster_list, desc="Padding vocab tensors"):

			jamo, wlctx, wrctx, elctx, erctx, _, _ = cluster.vocab_tensors

			if cluster.max_jamo > max_jamo:
				cut = []
				for i, item in enumerate(jamo):
					# print(len(item))
					if len(item) > max_jamo:
						cut.append(i)
				# print(cut)
				jamo = [x for i, x in enumerate(jamo) if i not in cut]
				wlctx = [x for i, x in enumerate(wlctx) if i not in cut]
				wrctx = [x for i, x in enumerate(wrctx) if i not in cut]
				elctx = [x for i, x in enumerate(elctx) if i not in cut]
				erctx = [x for i, x in enumerate(erctx) if i not in cut]

			for item in jamo:
				item += [0] * (max_jamo - len(item))
			if len(jamo) == 0:
				invalid_cluster.append(cluster)
				continue
			if len(jamo) < max_voca:
				pad = max_voca - len(jamo)
				jamo += [[0] * max_jamo] * pad
				wlctx += [[0] * self.ctx_window_size] * pad
				wrctx += [[0] * self.ctx_window_size] * pad
				elctx += [[0] * self.ctx_window_size] * pad
				erctx += [[0] * self.ctx_window_size] * pad
			elif len(jamo) > max_voca:
				jamo = jamo[:max_voca]
				wlctx = wlctx[:max_voca]
				wrctx = wrctx[:max_voca]
				elctx = elctx[:max_voca]
				erctx = erctx[:max_voca]

			assert len(jamo) == max_voca
			assert len(jamo[0]) == max_jamo
			cluster.cluster = cluster.cluster[:max_voca]
			cluster.update_tensor(*[torch.tensor(x) for x in [jamo, wlctx, wrctx, elctx, erctx]])
		print(len(invalid_cluster))
		corpus.cluster = {k: v for k, v in corpus.cluster.items() if v not in invalid_cluster}
		corpus.additional_cluster = [c for c in corpus.additional_cluster if c not in invalid_cluster]
	def convert_cluster_to_tensor(self, corpus, max_jamo_restriction):
		# for validation
		if type(corpus) is not Corpus:
			corpus = Corpus.load_corpus(corpus)
			print(len(corpus))
		if self.mode != "demo":
			self.mark_kb(corpus)
		self.generate_cluster_vocab_tensors(corpus, max_jamo_restriction=max_jamo_restriction)
		return corpus
