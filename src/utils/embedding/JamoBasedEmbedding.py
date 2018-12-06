import itertools
from .AbstractEmbedding import AbstractEmbedding
from .. import KoreanUtil, GenToIter
from gensim.models import Word2Vec, FastText, KeyedVectors
import numpy as np
import os
class JamoBasedEmbedding(AbstractEmbedding):
	def __init__(self, dim, base_fn):
		super().__init__(dim)
		target_fn = {"word2vec": Word2Vec, "fasttext": FastText}
		self.model_fn = target_fn[base_fn.lower()]

	@property
	def file_prefix(self):
		return "jamo"
	
	def train_embedding(self, corpus_name, corpus_generator, dim, window_size):
		dim += 3 - (dim % 3)
		corpus1, corpus2 = itertools.tee(corpus_generator)
		jc = map(KoreanUtil.decompose_sent, corpus1)
		self.jamo_wv = self.model_fn(jc, size=dim // 3, window=window_size, min_count=5).wv

		# corpus1, corpus2 = itertools.tee(corpus_generator)
		# jc1 = map(KoreanUtil.decompose_sent, corpus1)
		# jc2 = map(KoreanUtil.decompose_sent, corpus2)
		self.char_wv = self.model_fn(GenToIter.MakeIter(corpus2), size=dim, window=window_size, min_count=5).wv

		self.jamo_wv.save(self.save_path+"_".join([self.file_prefix, str(dim), corpus_name, "jamo"])+".bin")
		self.char_wv.save(self.save_path+"_".join([self.file_prefix, str(dim), corpus_name, "char"])+".bin")

	def load_embedding(self, corpus_name, dim, corpus_generator):
		if os.path.isfile(self.save_path+"_".join([self.file_prefix, str(dim), corpus_name, "jamo"])+".bin") and \
		   os.path.isfile(self.save_path+"_".join([self.file_prefix, str(dim), corpus_name, "char"])+".bin"):
			self.jamo_wv = KeyedVectors.load(self.save_path+"_".join([self.file_prefix, str(dim), corpus_name, "jamo"])+".bin")
			self.char_wv = KeyedVectors.load(self.save_path+"_".join([self.file_prefix, str(dim), corpus_name, "char"])+".bin")
			return
		if corpus_generator is None:
			raise Exception("To initialize new embedding, you should specify corpus generator")
		self.train_embedding(corpus_name, corpus_generator, dim, 5)

	def __getitem__(self, item):
		if item is not str or len(item) > 1: raise Exception("Must feed string")
		if KoreanUtil.is_korean_character(item):
			jamo_decomposed = KoreanUtil.char_to_elem(item)
			return np.concatenate(list(map(lambda x: self.jamo_wv[x], jamo_decomposed)))
		return char_wv[item]
		# 근데 이렇게 하면 그냥 character embedding을 구하고 싶을 때 좀 불리할듯. 역시 그냥 논문을 구현해야 한다.
		# 일단 이 모듈의 컨셉이라고 치자...