from ...ds.Corpus import Corpus
from ...ds.Vocabulary import Vocabulary
from ...utils import readfile, jsonload, jsondump, TimeUtil, split_to_batch, Embedding
import random
import numpy as np
import os
import logging
class DataGenerator():
	def __init__(self, args):
		logging.info("Initializing DataGenerator")
		# load embedding dict
		self.w2i, we = Embedding.load_embedding(args.word_embedding_path, args.word_embedding_type)
		self.e2i, ee = Embedding.load_embedding(args.entity_embedding_path, args.entity_embedding_type)
		# self.w2i = {w: i for i, w in enumerate(readfile(args.word_embedding_path+".word"))}
		# self.e2i = {e: i for i, e in enumerate(readfile(args.entity_embedding_path+".word"))}
		self.batch_size = args.batch_size
		# check if we can load data from pre-defined cluster
		if args.data_load_path is not None:
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
				self.change_into_tensor()
				return
			except:
				pass

		self.data_path = args.data_path
		self.fake_er_rate = args.fake_er_rate
		self.fake_el_rate = args.fake_el_rate
		self.fake_ec_rate = args.fake_ec_rate

		self.generate_data()

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
	def change_into_tensor(self):
		for k, cluster in self.corpus.clusters.items():
			for vocab in cluster:
				lctxw_ind = [self.w2i[x] if x in self.w2i else 0 for x in vocab.lctx]
				rctxw_ind = [self.w2i[x] if x in self.w2i else 0 for x in vocab.rctx]

				lctxe_ind = [self.e2i[x] if x in self.e2i else 0 for x in vocab.lctx_ent]
				rctxe_ind = [self.e2i[x] if x in self.e2i else 0 for x in vocab.rctx_ent]
				# near_words = [self.w2i[x] if x in self.w2i else 0 for x in (vocab.lctx + vocab.rctx)]
				# near_entities = [self.e2i[x] for x in (vocab.lctx_ent + vocab.rctx_ent) if x in self.e2i]

				# cluster.vocab_tensors.append((near_words, near_entities, cluster.id, vocab.error_type))
				cluster.vocab_tensors["lctx_words"].append(lctxw_ind)
				cluster.vocab_tensors["rctx_words"].append(rctxw_ind)
				cluster.vocab_tensors["lctx_entities"].append(lctxe_ind)
				cluster.vocab_tensors["rctx_entities"].append(rctxe_ind)
				cluster.vocab_tensors["error_type"].append(vocab.error_type)


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
				pass

	def get_cluster_batch(self):
		buf = []
		for cluster in self.cluster.values():
			for token in cluster:
				pass