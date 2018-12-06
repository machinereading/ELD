#######################
# Embedding generator #
#######################
from abc import ABC, abstractmethod, abstractproperty
import numpy as np
class AbstractEmbedding(ABC):
	def __init__(self, corpus_name):
		self.save_path = "word_embeddings/"
		
		self.load_embedding(corpus_name)

	@property
	def save_file_name(self):
		return self.file_prefix+"_"+str(self.dim)
	
	@abstractproperty
	def file_prefix(self):
		return None

	# @abstractmethod
	# def embedding_save_fn(self):
	# 	return None

	# @abstractmethod
	# def train_embedding(self, corpus_name, corpus_generator, window_size):
	# 	return None

	@abstractmethod
	def load_embedding(self, corpus_name):
		pass

	def __getitem__(self, item):
		return self.wv[item] if type(item) is str and item in self.wv else np.zeros([self.dim,], np.float32)