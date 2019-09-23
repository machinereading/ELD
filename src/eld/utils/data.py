from copy import deepcopy
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from .args import ELDArgs
from ...ds import *
from ...el.utils.data import CandDict
from ...utils import readfile, pickleload, KoreanUtil
import math
class ELDDataset(Dataset):
	def __init__(self, corpus: Corpus, args: ELDArgs):
		self.corpus = corpus
		self.max_character_len_in_character = max(map(len, self.corpus.token_iter()))
		# self.max_token_len_in_sentence = max(map(len, self.corpus))
		self.device = args.device
		self.window_size = args.context_window_size
		self.ce_dim = args.ce_dim
		self.we_dim = args.we_dim
		self.ee_dim = args.ee_dim

		self.ce_flag = args.use_character_embedding
		self.we_flag = args.use_word_context_embedding
		self.ee_flag = args.use_entity_context_embedding
		self.re_flag = args.use_relation_embedding
		self.te_flag = args.use_type_embedding

	def __getitem__(self, index):
		# return ce, cl, lcwe, lcwl, rcwe, rcwl, lcee, lcel, rcee, rcel, re, rl, te, tl, lab
		# entity에 대한 주변 단어? -> we와 ee는 context, te는 자기 type,
		pad = lambda tensor, dim, pad_size, tensor_length: torch.stack([*tensor] + [torch.zeros(dim) for _ in range(pad_size - tensor_length)])
		target = self.corpus[index]

		ce, lwe, rwe, lee, ree, re, te, lab = target.tensor
		cl = lwl = rwl = lel = rel = rl = tl = 0
		if self.ce_flag:
			cl = ce.size()[0]
			ce = pad(ce, self.ce_dim, self.max_character_len_in_character, cl)
		if self.we_flag:
			lwe = lwe[-self.window_size:]
			rwe = rwe[:self.window_size]
			lwl = lwe.size()[0]
			rwl = rwe.size()[0]
			lwe = pad(lwe, self.we_dim, self.window_size, lwl)
			rwe = pad(rwe, self.we_dim, self.window_size, rwl)
		if self.ee_flag:
			lee = lee[-self.window_size:]
			ree = ree[:self.window_size]
			lel = lee.size()[0]
			rel = ree.size()[0]
			lee = pad(lee, self.ee_dim, self.window_size, lel)
			ree = pad(ree, self.ee_dim, self.window_size, rel)
		if self.re_flag:
			rl = re.shape()[0]
		if self.te_flag:
			tl = te.shape()[0]

		return ce, cl, lwe, lwl, rwe, rwl, lee, lel, ree, rel, re, rl, te, tl, lab

	def __len__(self):
		return len([x for x in self.corpus.entity_iter()])

class DataModule:
	def __init__(self, mode: str, args: ELDArgs):
		# index 0: not in dictionary
		self.ce_flag = args.use_character_embedding
		self.we_flag = args.use_word_context_embedding
		self.ee_flag = args.use_entity_context_embedding
		self.re_flag = args.use_relation_embedding
		self.te_flag = args.use_type_embedding
		self.ent_list = [x for x in readfile(args.ent_list_path)]
		self.redirects = pickleload(args.redirects_path)
		self.surface_ent_dict = CandDict(self.ent_list, pickleload(args.entity_dict_path), self.redirects)

		self.e2i = {w: i + 1 for i, w in enumerate(readfile(args.entity_file))}
		self.i2e = {v: k for k, v in self.e2i.items()}
		ee = np.load(args.entity_embedding_file)
		self.entity_embedding = np.stack([np.zeros(ee.shape[-1]), *ee])
		self.ee_dim = args.ee_dim = ee.shape[-1]
		self.ce_dim = self.we_dim = self.re_dim = self.te_dim = 1
		if self.ce_flag:
			self.c2i = {w: i + 1 for i, w in enumerate({KoreanUtil.cho + KoreanUtil.jung + KoreanUtil.jong})}
			self.character_embedding = torch.nn.embedding(len(self.c2i), args.ce_dim, padding_idx=0, max_norm=math.sqrt(3 / args.ce_dim))
			self.ce_dim = args.ce_dim
		if self.we_flag:
			self.w2i = {w: i + 1 for i, w in enumerate(readfile(args.word_file))}
			we = np.load(args.word_embedding_file)
			self.word_embedding = np.stack([np.zeros(we.shape[-1]), *we])
			self.we_dim = args.we_dim = we.shape[-1]
		if self.re_flag:
			self.r2i = {w: i + 1 for i, w in enumerate(readfile(args.relation_file))}
			re = np.load(args.relation_embedding_file)
			self.relation_embedding = np.stack([np.zeros(re.shape[-1]), *re])
			self.re_dim = args.re_dim = re.shape[-1]
		if self.te_flag:
			self.t2i = {w: i + 1 for i, w in enumerate(readfile(args.type_file))}
			te = np.load(args.type_embedding_file)
			self.type_embedding = np.stack([np.zeros(te.shape[-1]), *te])
			self.te_dim = args.te_dim = te.shape[-1]

		if mode == "train":
			self.train_corpus = Corpus.load_corpus(args.train_corpus_dir)
			self.dev_corpus = Corpus.load_corpus(args.dev_corpus_dir)

			self.initialize_vocabulary_tensor(self.train_corpus)
			self.initialize_vocabulary_tensor(self.dev_corpus)

			self.train_dataset = ELDDataset(self.train_corpus, args)
			self.dev_dataset = ELDDataset(self.dev_corpus, args)
		else:
			self.corpus = Corpus.load_corpus(args.corpus_dir)
			self.initialize_vocabulary_tensor(self.corpus)
			self.test_dataset = ELDDataset(self.corpus, args)

	def initialize_vocabulary_tensor(self, corpus: Corpus):
		for sentence in corpus:
			for token in sentence:
				if self.ce_flag:
					token.char_embedding = np.stack(self.character_embedding[self.c2i[x] if x in self.c2i else 0] for x in token.jamo)
				if self.we_flag:
					token.word_embedding = self.word_embedding[self.w2i[token.surface]] if token.surface in self.w2i else np.zeros(self.we_dim, np.float)
				if self.ee_flag:
					token.entity_embedding = self.entity_embedding[token.entity] if token.is_entity and token.entity in self.e2i else np.zeros(self.ee_dim, np.float)
				if self.re_flag:
					token.relation_embedding = None
				if self.te_flag:
					token.type_embedding = None

	def postprocess(self, corpus: Corpus, entity_label: List, make_copy=False) -> Corpus:
		"""
		corpus와 entity label을 받아서 corpus 내의 entity로 판명난 것들에 대해 entity명 및 타입 부여
		:param corpus: corpus
		:param entity_label: list of int
		:param make_copy: True일 경우 corpus를 deepcopy해서 복사본을 만듬. 원본이 필요한 경우 사용
		:return: entity가 표시된 corpus
		"""

		if make_copy:
			corpus = deepcopy(corpus)
		for entity, label in zip(corpus.entity_iter(), entity_label):
			target_entity = self.i2e[label] if label > 0 else "_" + entity.surface.replace(" ", "_")
			entity.entity = target_entity
		return corpus
