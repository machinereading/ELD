from .AbstractEmbedding import AbstractEmbedding
import pickle

class Polyglot(AbstractEmbedding):
	@property
	def file_prefix(self):
		return ""

	def load_embedding(self, corpus_name):
		words, embeddings = pickle.load(open(self.save_path+"polyglot-ko.pkl", 'rb'), encoding="latin1")
		self.wv = {}
		for w, e in zip(words, embeddings):
			self.wv[w] = e

	def __getitem__(self, item):
		pass