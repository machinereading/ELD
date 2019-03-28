from .Sentence import Sentence
from .Cluster import Cluster
from ..utils import jsonload
from tqdm import tqdm
import logging
class Corpus():
	def __init__(self):
		self.corpus = [] # list of sentence
		self.cluster = {} # dict of str(entity form): Cluster

	def add_sentence(self, sentence):
		self.corpus.append(sentence)
		sentence.id = len(self.corpus)

	def __iter__(self):
		for item in self.corpus:
			yield item
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
			corpus.corpus.append(Sentence.from_json(sentence))
		for cluster in json["cluster"]:
			corpus.cluster[cluster["target_entity"]] = Cluster.from_json(cluster)

