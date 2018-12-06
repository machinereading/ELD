import itertools
from .AbstractEmbedding import AbstractEmbedding
from .. import KoreanUtil, GenToIter
from gensim.models import Word2Vec, FastText, KeyedVectors
import numpy as np
import os
class JamoEmbedding(AbstractEmbedding):
	def __init__(self, corpus_name, dim=180, base_fn="word2vec"):
		self.dim = dim
		super(JamoEmbedding, self).__init__(corpus_name)

		self.dim += [0,2,1][self.dim % 3]
		target_fn = {"word2vec": Word2Vec, "fasttext": FastText}
		self.model_fn = target_fn[base_fn.lower()]

	@property
	def file_prefix(self):
		return "jamo"


	def train_embedding(self, corpus_name, corpus_generator, window_size):
		
		corpus1, corpus2 = itertools.tee(corpus_generator)
		jc = map(KoreanUtil.decompose_sent, corpus1)
		# for item in jc:
		# 	print(item)

		self.jamo_wv = self.model_fn(jc, size=self.dim // 3, window=window_size, sg=1).wv

		# corpus1, corpus2 = itertools.tee(corpus_generator)
		# jc1 = map(KoreanUtil.decompose_sent, corpus1)
		# jc2 = map(KoreanUtil.decompose_sent, corpus2)
		self.char_wv = self.model_fn(GenToIter.MakeIter(corpus2), size=self.dim, window=window_size, min_count=5, sg=1).wv

		self.jamo_wv.save(self.save_path+"_".join([self.file_prefix, str(self.dim), corpus_name, "jamo"])+".bin")
		self.char_wv.save(self.save_path+"_".join([self.file_prefix, str(self.dim), corpus_name, "char"])+".bin")

	def load_embedding(self, corpus_name):
		if os.path.isfile(self.save_path+"_".join([self.file_prefix, str(self.dim), corpus_name, "jamo"])+".bin") and \
		   os.path.isfile(self.save_path+"_".join([self.file_prefix, str(self.dim), corpus_name, "char"])+".bin"):
			self.jamo_wv = KeyedVectors.load(self.save_path+"_".join([self.file_prefix, str(self.dim), corpus_name, "jamo"])+".bin")
			self.char_wv = KeyedVectors.load(self.save_path+"_".join([self.file_prefix, str(self.dim), corpus_name, "char"])+".bin")
			return True
		return False
		

	def __getitem__(self, item):
		if type(item) is not str or len(item) > 1: raise Exception("Must feed string")
		if KoreanUtil.is_korean_character(item):
			jamo_decomposed = KoreanUtil.char_to_elem(item)
			return np.concatenate(list(map(lambda x: self.jamo_wv[x], jamo_decomposed)))
		return self.char_wv[item] if item in self.char_wv else np.zeros([self.dim,], np.float32)
		# 근데 이렇게 하면 그냥 character embedding을 구하고 싶을 때 좀 불리할듯. 역시 그냥 논문을 구현해야 한다.
		# 일단 이 모듈의 컨셉이라고 치자...