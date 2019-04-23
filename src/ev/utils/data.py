from .Tokenizer import Tokenizer
from ...ds.Corpus import Corpus
from ...ds.Cluster import Cluster
from ...ds.Vocabulary import Vocabulary
from ...utils import readfile, jsonload, jsondump, TimeUtil, split_to_batch, KoreanUtil

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import random
import os
import logging
import traceback

class SentenceGenerator(Dataset):
	def __init__(self, corpus):
		self.corpus = corpus

	def __len__(self):
		return sum(self.corpus.tagged_voca_lens)

	def __getitem__(self, ind):
		voca = self.corpus[ind]
		return voca.lctxw_ind, voca.rctxw_ind, voca.lctxe_ind, voca.rctxe_ind, voca.error_type+1

class ClusterGenerator(Dataset):
	def __init__(self, corpus):
		self.corpus = corpus

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



class DataGenerator():
	def __init__(self, mode, args):
		logging.info("Initializing DataGenerator")
		# 0 for oov, 1 for out of range
		self.make_word_tensor = args.word_embedding_type == "glove"
		self.make_entity_tensor = args.entity_embedding_type == "glove"

		w2i = {w: i for i, w in enumerate(readfile(args.word_embedding_path+".word"))} if self.make_word_tensor else None
		self.wt = Tokenizer(args.word_embedding_type, w2i)
		self.wt_pad = len(w2i)+1

		e2i = {e: i for i, e in enumerate(readfile(args.entity_embedding_path+".word"))} if self.make_entity_tensor else None
		self.et = Tokenizer(args.entity_embedding_type, e2i)
		self.et_pad = len(e2i)+1
		self.chunk_size = args.chunk_size

		self.kb = readfile(args.kb)
		# check if we can load data from pre-defined cluster
		if mode == "train":
			self.batch_size = args.batch_size
			self.ctx_window_size = args.ctx_window_size
			self.filter_data_tokens = args.filter_data_tokens
			if args.data_load_path is not None:
				self.data_load_path = args.data_load_path
				corpus_path = args.data_load_path
				try:
					path = "/".join(corpus_path.split("/")[:-1])+"/"
					file_prefix = corpus_path.split("/")[-1]
					sentence = []
					cluster = []
					if len(os.listdir(path)) == 0:
						raise FileNotFoundError("No save file in dir %s" % corpus_path)
					for item in os.listdir(path):
						if not item.startswith(file_prefix): continue
						if "sentence" in item:
							sentence += jsonload(path+item)
						
					self.corpus = Corpus.from_json(sentence)
					# self.generate_vocab_tensors()
					self.generate_cluster_vocab_tensors(self.corpus, max_voca_restriction=getattr(args, "max_voca_restriction", -1), max_jamo_restriction=getattr(args, "max_jamo_restriction", -1))
					return
				except FileNotFoundError:
					traceback.print_exc()
				except:
					traceback.print_exc()
					import sys
					sys.exit(1)


			self.data_path = args.data_path
			self.fake_er_rate = args.fake_er_rate
			self.fake_el_rate = args.fake_el_rate
			self.fake_ec_rate = args.fake_ec_rate
			self.generate_data()
			# self.generate_vocab_tensors()
			self.generate_cluster_vocab_tensors()

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

	def add_fake_ec(self, sentence):
		pass

	def save(self, path):
		j = self.corpus.to_json()
		
		for i, item in enumerate(split_to_batch(j, 1000)):
			jsondump(item, path+"_sentence_%d.json" % i)

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
		# Deprecated: BERT need original tokenizer
		# corpus postprocessing
		logging.info("Generating Vocab tensors...")
		filter_size = 10
		for sentence in tqdm(self.corpus, desc="Generating vocab tensors", total = len(self.corpus.corpus)):
			# print(len(sentence))
			sentence.tagged_voca_len = 0
			er_error_tokens = 0
			el_error_tokens = 0
			dark_entity_tokens = 0
			for vocab in sentence:
				if self.filter_data_tokens:
					if er_error_tokens >= filter_size and not vocab.is_entity:
						continue 
					if el_error_tokens >= filter_size and vocab.is_entity and vocab.entity_in_kb:
						continue
					if dark_entity_tokens >= filter_size:
						continue
				lctxw_ind = self.wt(vocab.lctx[-10:])[-self.ctx_window_size:]
				vocab.lctxw_ind = torch.tensor([self.wt_pad for _ in range(self.ctx_window_size - len(lctxw_ind))] + lctxw_ind)

				rctxw_ind = self.wt(vocab.rctx[:10])[:self.ctx_window_size]
				vocab.rctxw_ind = torch.tensor(([self.wt_pad for _ in range(self.ctx_window_size - len(rctxw_ind))] + rctxw_ind)[::-1])

				lctxe_ind = self.et(vocab.lctx_ent[-10:])[-self.ctx_window_size:]
				vocab.lctxe_ind = torch.tensor([self.et_pad for _ in range(self.ctx_window_size - len(lctxe_ind))] + lctxe_ind)
				
				rctxe_ind = self.et(vocab.rctx_ent[:10])[:self.ctx_window_size]
				vocab.rctxe_ind = torch.tensor(([self.et_pad for _ in range(self.ctx_window_size - len(rctxe_ind))] + rctxe_ind)[::-1])
				# print(vocab.entity, lctxw_ind, rctxw_ind, lctxe_ind, rctxe_ind) # why everything is zero?
				sentence.tagged_voca_len += 1
				sentence.tagged_tokens.append(vocab)
				vocab.tagged = True
				if self.filter_data_tokens:
					if not vocab.is_entity:
						er_error_tokens += 1
					elif vocab.is_entity and vocab.entity_in_kb:
						el_error_tokens += 1
					else:
						dark_entity_tokens += 1
		self.corpus.tagged_voca_lens = [x.tagged_voca_len for x in self.corpus.corpus]

	@TimeUtil.measure_time
	def generate_cluster_vocab_tensors(self, corpus, max_voca_restriction=None, max_jamo_restriction=None):
		logging.info("Generating Vocab tensors...")
		for cluster in tqdm(corpus.cluster.values(), desc="Generating vocab tensors"):
			if cluster.target_entity not in self.kb:
				cluster.target_entity = None
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
		logging.info("Done")
		# add padding
		logging.info("Add padding...")
		max_voca = max([len(x) for x in corpus.cluster.values()])
		if max_voca_restriction is not None and max_voca_restriction > 0:
			max_voca = min(max_voca, max_voca_restriction)
		max_voca += self.chunk_size - max_voca % self.chunk_size if max_voca % self.chunk_size > 0 else 0
		
		max_jamo = max([x.max_jamo for x in corpus.cluster.values()])
		if max_jamo_restriction is not None and max_jamo_restriction > 0:
			max_jamo = min(max_jamo, max_jamo_restriction)
		# max_jamo += self.chunk_size - max_jamo % self.chunk_size if max_jamo % self.chunk_size > 0 else 0

		print("Max vocabulary in cluster(with padding):", max_voca)
		print("Max jamo in word: ", max_jamo)
		for cluster in tqdm(corpus.cluster.values(), desc="Padding vocab tensors"):
			jamo, wlctx, wrctx, elctx, erctx, _, _ = cluster.vocab_tensors
			if cluster.max_jamo > max_jamo:
				cut = []
				for i, item in enumerate(jamo):
					if len(jamo) > max_jamo_restriction:
						cut.append(i)
				jamo = [x for i, x in enumerate(jamo) if i not in cut]
				wlctx = [x for i, x in enumerate(wlctx) if i not in cut]
				wrctx = [x for i, x in enumerate(wrctx) if i not in cut]
				elctx = [x for i, x in enumerate(elctx) if i not in cut]
				erctx = [x for i, x in enumerate(erctx) if i not in cut]

			for item in jamo:
				item += [0] * (max_jamo - len(item))

			pad = max_voca - len(jamo)
			jamo += [[0] * max_jamo] * pad
			wlctx += [[0] * self.ctx_window_size] * pad
			wrctx += [[0] * self.ctx_window_size] * pad
			elctx += [[0] * self.ctx_window_size] * pad
			erctx += [[0] * self.ctx_window_size] * pad
			cluster.update_tensor(torch.tensor(jamo), torch.tensor(wlctx), torch.tensor(wrctx), torch.tensor(elctx), torch.tensor(erctx))
		logging.info("Done")


	def convert_cluster_to_tensor(self, j):
		corpus = Corpus.load_corpus(j, filter_nik=True)
		self.generate_cluster_vocab_tensors(corpus)
		return corpus