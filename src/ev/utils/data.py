from ...ds.Corpus import Corpus
from ...ds.Vocabulary import Vocabulary
from ...utils import readfile, jsonload, jsondump, TimeUtil, split_to_batch, Embedding

import numpy as np
import torch
from tqdm import tqdm

import random
import os
import logging

class SentenceGenerator(torch.utils.data.Dataset):
	def __init__(self, corpus):
		self.corpus = corpus

	def __len__(self):
		return sum(self.corpus.tagged_voca_lens)

	def __getitem__(self, ind):
		voca = self.corpus[ind]
		return voca.lctxw_ind, voca.rctxw_ind, voca.lctxe_ind, voca.rctxw_ind, voca.error_type+1

class ClusterGenerator(torch.utils.data.Dataset):
	def __init__(self, corpus):
		self.corpus = corpus

	def __len__(self):
		return len(self.corpus.cluster)

	def __getitem__(self, ind):
		return self.corpus[ind] # TODO


class DataGenerator():
	def __init__(self, args):
		logging.info("Initializing DataGenerator")
		# load embedding dict - need to change
		# self.w2i, we = Embedding.load_embedding(args.word_embedding_path, args.word_embedding_type)
		# self.e2i, ee = Embedding.load_embedding(args.entity_embedding_path, args.entity_embedding_type)
		# 0 for oov, 1 for out of range
		self.w2i = {w: i+2 for i, w in enumerate(readfile(args.word_embedding_path+".word"))}
		self.e2i = {e: i+2 for i, e in enumerate(readfile(args.entity_embedding_path+".word"))}
		self.batch_size = args.batch_size
		self.ctx_window_size = args.ctx_window_size
		self.filter_data_tokens = args.filter_data_tokens
		# check if we can load data from pre-defined cluster
		if args.data_load_path is not None:
			corpus_path = args.data_load_path
			try:
				path = "/".join(corpus_path.split("/")[:-1])+"/"
				file_prefix = corpus_path.split("/")[-1]
				sentence = []
				cluster = []
				for item in os.listdir(path):
					if not item.startswith(file_prefix): continue
					if "sentence" in item:
						sentence += jsonload(path+item)
					elif "cluster" in item:
						cluster += jsonload(path+item)
				self.corpus = Corpus.from_json({"sentence": sentence, "cluster": cluster})
				self.generate_vocab_tensors()
				return
			except FileNotFoundError:
				import traceback
				traceback.print_exc()
			except:
				import traceback
				traceback.print_exc()
				import sys
				sys.exit(1)


		self.data_path = args.data_path
		self.fake_er_rate = args.fake_er_rate
		self.fake_el_rate = args.fake_el_rate
		self.fake_ec_rate = args.fake_ec_rate
		self.generate_data()
		# self.generate_vocab_tensors()

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

	def extract_fake_tokens(self, sentence):
		for token in sentence:
			if not token.is_entity and len(token) >= 2 and random.random() < self.fake_er_rate:
				token.error_type = 0
				self.fake_tokens.append(token)
			if token.is_entity and token.entity_in_kb and random.random() < self.fake_el_rate:
				token.error_type = 1
				self.fake_tokens.append(token)

	def add_fake_ec(self, sentence):
		pass

	def save(self, path):
		j = self.corpus.to_json()
		s = j["sentence"]
		c = j["cluster"]
		for i, item in enumerate(split_to_batch(s, 1000)):
			jsondump(item, path+"_sentence_%d.json" % i)
		for i, item in enumerate(split_to_batch(c, 1000)):
			jsondump(item, path+"_cluster_%d.json" % i)

	@classmethod
	def from_predefined_cluster(cls, corpus_path):
		gen = DataGenerator(None, init=False)
		path = "/".join(corpus_path.split("/")[:-1])+"/"
		file_prefix = corpus_path.split("/")[-1]
		sentence = []
		cluster = []
		for item in os.listdir(path):
			if not item.startswith(file_prefix): continue
			if "sentence" in item:
				sentence += jsonload(path+item)
			elif "cluster" in item:
				cluster += jsonload(path+item)
		gen.corpus = Corpus.from_json({"sentence": sentence, "cluster": cluster})
		
		return gen

	@TimeUtil.measure_time
	def generate_vocab_tensors(self):
		# corpus postprocessing
		logging.info("Generating Vocab tensors...")
		# print(len(self.w2i), len(self.e2i))
		for sentence in tqdm(self.corpus, desc="Generating vocab tensors", total = len(self.corpus.corpus)):
			# print(len(sentence))
			sentence.tagged_voca_len = 0
			er_error_tokens = 0
			el_error_tokens = 0
			dark_entity_tokens = 0
			for vocab in sentence:
				if self.filter_data_tokens:
					if er_error_tokens >= 10 and not vocab.is_entity:
						continue 
					if el_error_tokens >= 10 and vocab.is_entity and vocab.entity_in_kb:
						continue
					if dark_entity_tokens >= 10:
						continue
				lctxw_ind = [self.w2i[x.surface] if x.surface in self.w2i else 0 for x in vocab.lctx[-self.ctx_window_size:]]
				vocab.lctxw_ind = torch.tensor([0 for _ in range(self.ctx_window_size - len(lctxw_ind))] + lctxw_ind)

				rctxw_ind = [self.w2i[x.surface] if x.surface in self.w2i else 0 for x in vocab.rctx[:self.ctx_window_size]]
				vocab.rctxw_ind = torch.tensor(([0 for _ in range(self.ctx_window_size - len(rctxw_ind))] + rctxw_ind)[::-1])

				lctxe_ind = [self.e2i[x.entity] if x.entity in self.e2i else 0 for x in vocab.lctx_ent[-self.ctx_window_size:]]
				vocab.lctxe_ind = torch.tensor([0 for _ in range(self.ctx_window_size - len(lctxe_ind))] + lctxe_ind)
				
				rctxe_ind = [self.e2i[x.entity] if x.entity in self.e2i else 0 for x in vocab.rctx_ent[:self.ctx_window_size]]
				vocab.rctxe_ind = torch.tensor(([0 for _ in range(self.ctx_window_size - len(rctxe_ind))] + rctxe_ind)[::-1])
				# print(vocab.entity, lctxw_ind, rctxw_ind, lctxe_ind, rctxe_ind) # why everything is zero?
				sentence.tagged_voca_len += 1
				sentence.tagged_tokens.append(vocab)
				if self.filter_data_tokens:
					if not vocab.is_entity:
						er_error_tokens += 1
					elif vocab.is_entity and vocab.entity_in_kb:
						el_error_tokens += 1
					else:
						dark_entity_tokens += 1
		self.corpus.tagged_voca_lens = [x.tagged_voca_len for x in self.corpus.corpus]
				

	def get_tensor_batch(self):
		buf = {"lctx_words": [],
			"rctx_words": [],
			"lctx_entities": [],
			"rctx_entities": [],
			"error_type": []}
		ind = 0
		for _, cluster in self.corpus.clusters.items():
			for k, v in cluster.vocab_tensors.items():
				buf[k].append(v)
				ind += 1
				if ind == self.batch_size:
					yield buf
					buf = \
						{
							"lctx_words": [],
							"rctx_words": [],
							"lctx_entities": [],
							"rctx_entities": [],
							"error_type": []
						}
					ind = 0

	def get_token_batch(self):
		buf = []
		for sentence in self.corpus:
			for token in sentence:
				if token.lctxe_ind is not None:
					# print(token.surface, token.is_entity, token.entity_in_kb)
					buf.append(token)
				if len(buf) == self.batch_size:
					yield buf
					buf = []

	def get_cluster_batch(self):
		buf = []
		for cluster in self.cluster.values():
			for token in cluster:
				buf.append(token)
				if len(buf) == self.batch_size:
					yield buf
					buf = []

