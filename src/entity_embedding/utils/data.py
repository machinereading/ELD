from ...utils import jsonload, KoreanUtil, dictload
from ...ds import Corpus
from torch.utils.data import Dataset, DataLoader
import torch
import os
import random
from typing import Dict, Iterable



class DataModule:
	# 굳이 새로 만들 필요가 있나?
	def __init__(self, corpus: Corpus, w2i: Dict, e2i: Dict):
		self.tokenizer = KoreanUtil.tokenize
		self.w2i = w2i
		self.e2i = e2i
		self.corpus = corpus
		self.tensorize()

	def tensorize(self):
		for sentence in self.corpus:
			for token in sentence:
				token.wi = dictload(token.surface, self.w2i)
				if token.is_entity:
					token.ei = dictload(token.entity, self.e2i)
			for token in sentence:
				token.lctxw_ind = [x.wi for x in token.lctx]
				token.rctxw_ind = [x.wi for x in token.rctx]
				token.lctxe_ind = [x.ei for x in token.lctx_ent]
				token.rctxe_ind = [x.ei for x in token.rctx_ent]

class EmbeddingDataset(Dataset):
	def __init__(self, data: DataModule):
		self.data = data
		self.ld = {}
		lbuf = 0
		for sentence in self.data.corpus:
			self.ld[lbuf] = sentence
			lbuf += len(sentence.kb_entities)
		self.maxlen = lbuf
		self.window_size = 5
	def __getitem__(self, index):
		lt = 0
		sin = 0
		for k, v in self.ld.items():
			if k > index:
				break
			lt = v
			sin = k
		token = lt.kb_entities[index - sin]
		wrapper = lambda target, direction: torch.LongTensor(EmbeddingDataset.set_len(target, self.window_size, direction))
		return wrapper(token.lctxw_ind, -1), \
			   wrapper(token.rctxw_ind, 1), \
			   wrapper(token.lctxe_ind, -1), \
			   wrapper(token.rctxe_ind, 1), \
			   token.ei

	def __len__(self):
		return self.maxlen

	@staticmethod
	def set_len(target, length: int, direction: int):
		"""

		:param target: target list to slice or pad
		:param length: set size
		:param direction: +1 or -1. +1 means -> direction - padding will go right side, -1 is opposite
		:return:
		"""
		assert direction in [1, -1]
		s = slice(None, length, None) if direction == 1 else slice(-length, None, None)
		target = target[s]
		if len(target) < length:
			pad = [0] * (length - len(target))
			target = target + pad if direction == 1 else pad + target
		return target
