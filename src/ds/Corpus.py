import logging

from tqdm import tqdm

from .Cluster import Cluster
from .Sentence import Sentence
from ..utils import jsonload, TimeUtil

class Corpus:
	def __init__(self):
		self.corpus = []  # list of sentence
		self.tagged_voca_lens = []
		self.cluster = {}  # dict of str(entity form): Cluster
		self.additional_cluster = []

	def add_sentence(self, sentence):
		self.corpus.append(sentence)
		sentence.id = len(self.corpus)

	def __iter__(self):
		for item in self.corpus:
			yield item

	def __len__(self):
		return len(self.corpus)

	@property
	def cluster_list(self):
		return [x for x in self.cluster.values()] + self.additional_cluster

	@property
	def id2c(self):
		return {i: v for i, v in enumerate(self.cluster_list)}

	@property
	def max_jamo(self):
		return max([x.max_jamo for x in self.cluster_list])

	@TimeUtil.measure_time
	def __getitem__(self, ind):
		acclen = 0
		accbuf = 0
		senind = 0
		for l in self.tagged_voca_lens:
			acclen += l
			if acclen > ind:
				return self.corpus[senind][ind - acclen]
			senind += 1
			accbuf += l
		raise IndexError

	def get_cluster_by_index(self, ind):
		return self.cluster_list[ind]

	@classmethod
	def load_corpus(cls, path):
		# load from crowdsourcing form
		if type(path) is str:
			path = jsonload(path)
		assert type(path) is list
		logging.info("Loading corpus")
		corpus = cls()
		for item in tqdm(path, desc="Loading corpus"):
			sentence = Sentence.from_cw_form(item)
			if sentence is None: continue
			if len(sentence.entities) == 0: continue
			corpus.add_sentence(sentence)

			for nt in sentence.entities:
				if nt.entity not in ["NOT_IN_CANDIDATE", "EMPTY_CANDIDATES", "NOT_AN_ENTITY"]:  # for gold set test
					entity = nt.entity
				else:
					entity = nt.surface
				if entity not in corpus.cluster:
					c = Cluster(entity)
					c.id = len(corpus.cluster)
					corpus.cluster[entity] = c
				corpus.cluster[entity].add_elem(nt)
		# corpus.id2c = {i: v for i, v in enumerate(corpus.cluster_list)}
		return corpus

	def to_json(self):
		return [sent.to_json() for sent in self.corpus]

	@classmethod
	def from_json(cls, json):
		if type(json) is str:
			json = jsonload(json)
		corpus = cls()
		for sentence in tqdm(json[:100], desc="Loading EV corpus"):  # limit data for runnablity
			sentence = Sentence.from_json(sentence)
			if len(sentence.entities) > 0:
				corpus.corpus.append(sentence)
			for token in sentence.entities:
				if token.entity not in corpus.cluster:
					corpus.cluster[token.entity] = Cluster(token.entity)
				corpus.cluster[token.entity].add_elem(token)
		# for cluster in json["cluster"]:
		# 	corpus.cluster[cluster["target_entity"]] = Cluster.from_json(cluster)
		# 	for token in cluster:
		# 		parent_sentence_id = token.parent_sentence
		# 		token.parent_sentence = corpus.corpus[parent_sentence_id]
		# 		assert token.parent_sentence.id == parent_sentence_id
		# corpus.id2c = {i: v for i, v in enumerate(corpus.cluster_list)}
		return corpus

	def split_sentence_to_dev(self):
		train = Corpus()
		dev = Corpus()
		for i, (sent, tlen) in enumerate(zip(self.corpus, self.tagged_voca_lens)):
			if i % 10 == 0:
				dev.corpus.append(sent)
				dev.tagged_voca_lens.append(tlen)
			else:
				train.corpus.append(sent)
				train.tagged_voca_lens.append(tlen)
		return train, dev

	def split_cluster_to_dev(self):
		train = Corpus()
		dev = Corpus()
		for i, (k, v) in enumerate(self.cluster.items()):
			if i % 10 == 0:
				dev.cluster[k] = v
			else:
				train.cluster[k] = v
		for i, v in enumerate(self.additional_cluster):
			if i % 10 == 0:
				dev.additional_cluster.append(v)
			else:
				train.additional_cluster.append(v)

		return train, dev
