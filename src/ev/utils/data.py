from ...ds.Corpus import Corpus
from ...ds.Vocabulary import Vocabulary
from ...utils import readfile, jsonload, jsondump, TimeUtil, split_to_batch
import random
import numpy as np
import os
class DataGenerator():
	def __init__(self, args):
		
		# load embedding dict
		self.w2i = {w: i for i, w in enumerate(readfile(args.word_embedding_path+".word"))}
		self.e2i = {e: i for i, e in enumerate(readfile(args.entity_embedding_path+".word"))}

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
		self.vocab_tensors = []
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
		
		gen.vocab_tensors = []
		return gen

	@TimeUtil.measure_time
	def change_into_tensor(self):
		for k, cluster in self.corpus.clusters.items():
			for vocab in cluster:
				near_words = [self.w2i[x] if x in self.w2i else np.zeros([0]) for x in (vocab.lctx + vocab.rctx)]
				near_entities = [self.e2i[x] for x in (vocab.lctx_ent + vocab.rctx_ent) if x in self.e2i]
				cluster.vocab_tensors.add((near_words, near_entities, cluster.id, vocab.error_type))


	def get_tensor_batch(self):
		for _, cluster in self.corpus.clusters.items():
			yield cluster.vocab_tensors