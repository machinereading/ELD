import os
from typing import Iterator, List

from tqdm import tqdm

from .Cluster import Cluster
from .Sentence import Sentence
from .Vocabulary import Vocabulary
from ..utils import jsonload, TimeUtil, diriter

class Corpus:
	def __init__(self):
		self.sentences = []  # list of sentence
		self.tagged_voca_lens = []  # For Iterative
		self.cluster = {}  # dict of str(entity form): Cluster --> for Clustering, Iterative
		self.additional_cluster = []  # For Iterative
		self._eld_items: List[Vocabulary] = []  # for ELD

	def add_sentence(self, sentence: Sentence):
		"""
		sentence를 추가하고 sentence에 id 부여
		@param sentence: 추가하고자 하는 sentence 개체
		@return: None
		"""
		self.sentences.append(sentence)
		sentence.id = len(self.sentences)

	def __iter__(self) -> Iterator[Sentence]:
		"""
		for문 돌릴 수 있게 만드는 것

		Sentence iteration을 돌림.
		@return: sentence iterator
		"""
		for item in self.sentences:
			yield item

	def __len__(self):
		return len(self.sentences)

	@property
	def cluster_list(self):
		return [x for x in self.cluster.values()] + self.additional_cluster

	@property
	def id2c(self):  # For Iterative
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
	def load_corpus(cls, path, limit: int = 0, min_token: int = 0):
		"""
		path에 있는 파일(들)에서 corpus를 뽑아내는 것. 이 때 path의 파일(들)은 crowdsourcing form이어야 함.
		@param path: 다음 중 하나.
			- list of dict: crowdsourcing form의 list
			- str: 단일 파일을 가리키는 위치 또는 파일들이 들어있는 디렉토리 위치
		@param limit: default: 0. 1 이상의 정수일 경우 앞에서부터 limit까지만 읽어옴
		@param min_token: default: 0. 문장의 token이 min_token 이하인 경우 자름
		@return: Crowdsourcing form을 읽어낸 Corpus object
		"""
		# load from crowdsourcing form
		if type(path) is str:
			if os.path.isfile(path):
				try:
					path = jsonload(path)
				except PermissionError or IsADirectoryError:
					try:
						if path[-1] != "/": path += "/"
						path = [jsonload(path + f) for f in os.listdir(path)]
					except:
						raise Exception("Data format error")
			else:
				path = [x for x in map(jsonload, diriter(path))]
		assert type(path) is list
		# logging.info("Loading corpus")
		corpus = cls()
		if limit > 0:
			path = path[:limit]
		for item in tqdm(path, desc="Loading corpus"):
			sentence = Sentence.from_cw_form(item)
			if sentence is None:
				# print("Sentence is None")
				continue
			if len(sentence.entities) == 0:
				# print("No entities")
				continue
			if min_token > 0 and len(sentence) < min_token: continue
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

	def to_json(self): # 안씀
		return [sent.to_json() for sent in self.sentences]

	@classmethod
	def from_json(cls, json): # 안씀
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
		"""
		Corpus 내의 sentence를 9:1 로 잘라내어 train corpus와 dev corpus로 분리.
		아마 안 쓸 듯?
		@return: train corpus, dev corpus
		"""
		train = Corpus()
		dev = Corpus()
		for i, sent in enumerate(self.sentences):
			if i % 10 == 0:
				dev.add_sentence(sent)
			else:
				train.add_sentence(sent)
		return train, dev

	def split_cluster_to_dev(self): # For Iterative
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

	def recluster(self): # For Iterative
		self.cluster = {}
		for sentence in self:
			for token in sentence.entities:
				if not hasattr(token, "ec_cluster"): continue
				if token.ec_cluster not in self.cluster:
					self.cluster[token.ec_cluster] = Cluster(str(token.ec_cluster))
				self.cluster[token.ec_cluster].add_elem(token)

	@property
	def token_len(self): # 전체 token 길이
		return sum(map(len, self))

	def token_iter(self):
		for sent in self:
			for token in sent:
				yield token

	def _entity_iter(self):
		for sent in self:
			for ent in sent.entities:
				yield ent

	# for ELD
	@property
	def eld_len(self):
		return len(self.eld_items)

	def eld_get_item(self, idx):
		if type(idx) is int:
			if idx > self.eld_len: raise IndexError(idx, self.eld_len)

			return self.eld_items[idx]
		elif type(idx) is slice:
			return self.eld_items[idx]

	@property
	def eld_items(self) -> List[Vocabulary]:
		return [x for x in self._entity_iter() if x.target]

	@property
	def entities(self):
		return [x for x in self._entity_iter()]

	@classmethod
	def from_string(cls, *corpus):
		from ..utils.datafunc import text_to_etri, etri_to_ne_dict
		return cls.load_corpus(list(map(etri_to_ne_dict, map(text_to_etri, corpus))))
