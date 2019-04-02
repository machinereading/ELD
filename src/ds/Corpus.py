from .Sentence import Sentence
from .Cluster import Cluster
from ..utils import jsonload
from tqdm import tqdm
import logging
class Corpus():
	def __init__(self):
		self.corpus = [] # list of sentence
		self.tagged_voca_lens = []
		self.cluster = {} # dict of str(entity form): Cluster

	def add_sentence(self, sentence):
		self.corpus.append(sentence)
		sentence.id = len(self.corpus)

	def __iter__(self):
		for item in self.corpus:
			yield item

	def __len__(self):
		return len(self.corpus)

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
		raise IndexOutOfRangeException

	@classmethod
	def load_corpus(cls, path):
		# load from crowdsourcing form
		if type(path) is str:
			path = jsonload(path)
		assert type(path) is list
		logging.info("Loading corpus")
		corpus = Corpus()
		for item in tqdm(path, desc="Loading corpus"):
			sentence = Sentence.from_cw_form(item)
			if sentence is None: continue
			if len(sentence.entities) == 0: continue
			corpus.add_sentence(sentence)
			for nt in sentence.not_in_kb_entities:
				if nt.entity not in corpus.cluster:
					c = Cluster()
					c.id = len(corpus.cluster)
					corpus.cluster[nt.entity] = c

				corpus.cluster[nt.entity].add_elem(nt)

		return corpus


	def to_json(self):
		return {
			"sentence": [sent.to_json() for sent in self.corpus],
			"cluster": [cluster.to_json() for cluster in self.cluster.values()]
		}
	@classmethod
	def from_json(cls, json):
		if type(json) is str:
			json = jsonload(json)
		corpus = Corpus()
		for sentence in json["sentence"]:
			sentence = Sentence.from_json(sentence)
			if len(sentence.entities) > 0:
				corpus.corpus.append(sentence)
		for cluster in json["cluster"]:
			corpus.cluster[cluster["target_entity"]] = Cluster.from_json(cluster)

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