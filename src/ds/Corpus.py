import logging
import os
from tqdm import tqdm

from .Cluster import Cluster
from .Sentence import Sentence
from .Vocabulary import Vocabulary
from ..utils import jsonload, TimeUtil, diriter
from typing import Iterator, List


class Corpus:
	def __init__(self):
		self.sentences = []  # list of sentence
		self.tagged_voca_lens = []
		self.cluster = {}  # dict of str(entity form): Cluster
		self.additional_cluster = []
		self._eld_items: List[Vocabulary] = []

	def add_sentence(self, sentence):
		self.sentences.append(sentence)
		sentence.id = len(self.sentences)

	def __iter__(self) -> Iterator[Sentence]:
		for item in self.sentences:
			yield item

	def __len__(self):
		return len(self.sentences)

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
	def __getitem__(self, ind) -> Vocabulary:
		acclen = 0
		accbuf = 0
		senind = 0
		for l in self.tagged_voca_lens:
			acclen += l
			if acclen > ind:
				return self.sentences[senind][ind - acclen]
			senind += 1
			accbuf += l
		raise IndexError

	def get_cluster_by_index(self, ind):
		return self.cluster_list[ind]

	@classmethod
	def load_corpus(cls, path):
		# load from crowdsourcing form
		if type(path) is str:
			if os.path.isfile(path):
				try:
					path = jsonload(path)
				except PermissionError or IsADirectoryError:
					try:
						if path[-1] != "/": path += "/"
						path = [jsonload(path+f) for f in os.listdir(path)]
					except:
						raise Exception("Data format error")
			else:
				path = [x for x in map(jsonload, diriter(path))]
		assert type(path) is list
		logging.info("Loading corpus")
		corpus = cls()
		for item in tqdm(path[:500], desc="Loading corpus"):
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
		return [sent.to_json() for sent in self.sentences]

	@classmethod
	def from_json(cls, json):
		if type(json) is str:
			json = jsonload(json)
		corpus = cls()
		for sentence in tqdm(json, desc="Loading EV corpus"):  # limit data for runnablity
			sentence = Sentence.from_json(sentence)
			if len(sentence.entities) > 0:
				corpus.sentences.append(sentence)
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
		for i, sent in enumerate(self.sentences):
			if i % 10 == 0:
				dev.add_sentence(sent)
			else:
				train.add_sentence(sent)
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

	def recluster(self):
		self.cluster = {}
		for sentence in self:
			for token in sentence.entities:
				if not hasattr(token, "ec_cluster"): continue
				if token.ec_cluster not in self.cluster:
					self.cluster[token.ec_cluster] = Cluster(str(token.ec_cluster))
				self.cluster[token.ec_cluster].add_elem(token)

	def token_iter(self):
		for sent in self:
			for token in sent:
				yield token

	def entity_iter(self):
		for sent in self:
			for ent in sent.entities:
				yield ent

	@property
	def eld_len(self):
		return len([x for x in self.entity_iter() if x.target])

	def eld_get_item(self, idx):
		if idx > self.eld_len: raise IndexError(idx, self.eld_len)

		return self.eld_items[idx]

	@property
	def eld_items(self) -> List[Vocabulary]:
		if len(self._eld_items) == 0:
			for ent in self.entity_iter():
				if ent.target:
					self._eld_items.append(ent)
		return self._eld_items

