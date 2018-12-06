from .AbstractEmbedding import AbstractEmbedding
from gensim.models import KeyedVectors
class PretrainedEmbedding(AbstractEmbedding):
	def __init__(self, embedding_path):
		super(PretrainedEmbedding, self).__init__(embedding_path)
		self.dim = self.wv.vector_size

	def file_prefix(self):
		return None

	def embedding_save_fn(self):
		return None

	def train_embedding(self, corpus_name, corpus_generator, dim, window_size):
		return None

	def load_embedding(self, embedding_path):
		self.wv = KeyedVectors.load_word2vec_format(self.save_path + embedding_path, binary=True)
		return True