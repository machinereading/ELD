from copy import deepcopy
import os

import math
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from typing import List

from src.utils import KoreanUtil
from . import flags
from .args import ELDArgs
from ...ds import *
from ...utils import readfile, pickleload, TimeUtil, one_hot
from ... import GlobalValues as gl

class ELDDataset(Dataset):
	def __init__(self, corpus: Corpus, args: ELDArgs):
		if not flags.data_initialized:
			raise Exception("Tensor not initialized")
		self.corpus = corpus
		self.max_jamo_len_in_word = args.jamo_limit
		self.max_word_len_in_entity = args.word_limit
		self.max_token_len_in_sentence = max(map(len, self.corpus))
		self.device = args.device
		self.window_size = args.context_window_size

		self.ce_dim = args.c_emb_dim
		self.we_dim = args.w_emb_dim
		self.ee_dim = args.e_emb_dim
		self.re_dim = args.r_emb_dim

		self.ce_flag = args.use_character_embedding
		self.we_flag = args.use_word_embedding
		self.wce_flag = args.use_word_context_embedding
		self.ee_flag = args.use_entity_context_embedding
		self.re_flag = args.use_relation_embedding
		self.te_flag = args.use_type_embedding

		self.r_limit = args.relation_limit

	# initialize maximum

	@TimeUtil.measure_time
	def __getitem__(self, index):
		# return ce, cl, lcwe, lcwl, rcwe, rcwl, lcee, lcel, rcee, rcel, re, rl, te, tl, lab
		# entity에 대한 주변 단어? -> we와 ee는 context, re는 다른 개체와의 관계, te는 자기 type
		def pad(tensor, pad_size, emb_dim):
			if tensor.dim() == 1:
				tensor = tensor.unsqueeze(0)
			return F.pad(tensor, [0, 0, 0, pad_size - tensor.size()[0]])
			# return torch.cat((tensor, torch.zeros(pad_size - tensor.size()[0], emb_dim, dtype=torch.float64)))
		target = self.corpus.eld_get_item(index)

		ce, we, lwe, rwe, lee, ree, re, te, lab = target.tensor
		cl = wl = lwl = rwl = lel = rel = rl = tl = 0
		if self.ce_flag:
			cl = ce.size()[0]
			ce = pad(ce, self.max_jamo_len_in_word, self.ce_dim)
		if self.we_flag:
			wl = we.size()[0]
			we = pad(we, self.max_word_len_in_entity, self.we_dim)
		if self.wce_flag:
			if len(lwe) == 0:
				lwe = [torch.zeros(1, self.we_dim, dtype=torch.float)]
			if len(rwe) == 0:
				rwe = [torch.zeros(1, self.we_dim, dtype=torch.float)]
			lwe = torch.cat(lwe, dim=0).view(-1, self.we_dim)[-self.window_size:]
			rwe = torch.cat(rwe, dim=0).view(-1, self.we_dim)[:self.window_size]
			lwl = lwe.size()[0]
			rwl = rwe.size()[0]
			lwe = pad(lwe, self.window_size, self.we_dim)
			rwe = pad(rwe, self.window_size, self.we_dim)
		if self.ee_flag:
			if len(lee) == 0:
				lee = [torch.zeros(1, self.ee_dim, dtype=torch.float)]
			if len(ree) == 0:
				ree = [torch.zeros(1, self.ee_dim, dtype=torch.float)]
			lee = torch.cat(lee, dim=0).view(-1, self.ee_dim)[-self.window_size:]
			ree = torch.cat(ree, dim=0).view(-1, self.ee_dim)[:self.window_size]
			lel = lee.size()[0]
			rel = ree.size()[0]
			lee = pad(lee, self.window_size, self.ee_dim)
			ree = pad(ree, self.window_size, self.ee_dim)
		if self.re_flag:
			re = target.relation_embedding
			rl = re.size()[0]
			re = pad(re, self.r_limit, self.re_dim)
		if self.te_flag:
			te = target.type_embedding
			tl = te.size()[0]

		return ce, cl, we, wl, lwe, lwl, rwe, rwl, lee, lel, ree, rel, re, rl, te, tl, lab

	def __len__(self):
		return self.corpus.eld_len

class DataModule:
	def __init__(self, mode: str, args: ELDArgs):
		# index 0: not in dictionary
		# initialize embedding
		self.ce_flag = args.use_character_embedding
		self.we_flag = args.use_word_context_embedding
		self.ee_flag = args.use_entity_context_embedding
		self.re_flag = args.use_relation_embedding
		self.te_flag = args.use_type_embedding
		self.ent_list = [x for x in readfile(args.ent_list_path)]
		self.redirects = pickleload(args.redirects_path)
		self.surface_ent_dict = CandDict(self.ent_list, pickleload(args.entity_dict_path), self.redirects)

		self.e2i = {w: i + 1 for i, w in enumerate(readfile(args.entity_file))}
		self.oe2i = {w: -i-1 for i, w in enumerate(readfile(args.out_kb_entity_file))}
		self.i2e = {v: k for k, v in self.e2i.items()}
		ee = np.load(args.entity_embedding_file)
		self.entity_embedding = torch.nn.Embedding.from_pretrained(torch.tensor(np.stack([np.zeros(ee.shape[-1]), *ee])).float())
		self.ee_dim = args.e_emb_dim = ee.shape[-1]
		self.ce_dim = self.we_dim = self.re_dim = self.te_dim = 1
		if self.ce_flag:
			self.c2i = {w: i + 1 for i, w in enumerate(readfile(args.character_file))}
			ce = np.load(args.character_embedding_file)
			self.character_embedding = torch.nn.Embedding.from_pretrained(torch.tensor(np.stack([np.zeros(ce.shape[-1]), *ce])).float())
			self.ce_dim = args.c_emb_dim = ce.shape[-1]
		if self.we_flag:
			self.w2i = {w: i + 1 for i, w in enumerate(readfile(args.word_file))}
			we = np.load(args.word_embedding_file)
			self.word_embedding = torch.nn.Embedding.from_pretrained(torch.tensor(np.stack([np.zeros(we.shape[-1]), *we])).float())
			self.we_dim = args.w_emb_dim = we.shape[-1]
		if self.re_flag:
			# relation embedding 방법
			# 1. [relation id, 상대적 position, incoming/outgoing, score] -> BATCH * WORD * (RELATION + 3) <- 이거로
			# 2. [incoming relation score, outgoing relation score, ...] -> BATCH * (WORD * 2)
			self.r2i = {w: i for i, w in enumerate(readfile(args.relation_file))}
			self.re_dim = args.r_emb_dim = len(self.r2i) + 3
			self.relation_limit = args.relation_limit

		if self.te_flag:  # one-hot?
			self.t2i = {w: i + 1 for i, w in enumerate(readfile(args.type_file))}
			self.te_dim = args.t_emb_dim = len(self.t2i)

		# load corpus and se
		if mode == "train":
			self.train_corpus = Corpus.load_corpus(args.train_corpus_dir)
			self.dev_corpus = Corpus.load_corpus(args.dev_corpus_dir)

			self.initialize_vocabulary_tensor(self.train_corpus, register_new_entity=True)
			gl.logger.info("Train corpus initialized")
			self.initialize_vocabulary_tensor(self.dev_corpus)
			gl.logger.info("Dev corpus initialized")
			flags.data_initialized = True
			self.train_dataset = ELDDataset(self.train_corpus, args)
			gl.logger.debug("Train corpus size: %d" % len(self.train_dataset))
			self.dev_dataset = ELDDataset(self.dev_corpus, args)
			gl.logger.debug("Dev corpus size: %d" % len(self.dev_dataset))
			args.jamo_limit = self.train_dataset.max_jamo_len_in_word
			args.word_limit = self.train_dataset.max_word_len_in_entity
		else:
			self.corpus = Corpus.load_corpus(args.corpus_dir)
			self.initialize_vocabulary_tensor(self.corpus)
			flags.data_initialized = True
			self.test_dataset = ELDDataset(self.corpus, args)
		self.new_entity_count = 0



	def initialize_vocabulary_tensor(self, corpus: Corpus, register_new_entity: bool = False):
		if register_new_entity:
			new_ents = set([])
			for token in corpus.eld_items:
				if token.entity not in self.e2i:
					new_ents.add(token.entity)
			ent_emb = torch.randn([len(new_ents), self.ee_dim], dtype=torch.float) # TODO 일단 random
			self.entity_embedding = torch.nn.Embedding.from_pretrained(torch.cat((self.entity_embedding.weight, ent_emb), dim=0).float())
			for ent in new_ents:
				self.e2i[ent] = len(self.e2i)
		for token in corpus.token_iter():
			if token.is_entity:
				token.entity_embedding = self.entity_embedding(torch.tensor(self.e2i[token.entity] if token.entity in self.e2i else 0))
			if self.ce_flag:
				token.char_embedding = self.character_embedding(torch.tensor([self.c2i[x] if x in self.c2i else 0 for x in token.jamo]))
			if self.we_flag:
				words = KoreanUtil.tokenize(token.surface)
				token.word_embedding = self.word_embedding(torch.tensor([self.w2i[x] if x in self.w2i else 0 for x in words]))
			if token.target:
				if self.re_flag:
					relations = []
					for rel in token.relation:
						x = one_hot(self.r2i[rel.relation], len(self.r2i))
						x.append(rel.relative_index)
						x.append(rel.score)
						x.append(1 if rel.outgoing else -1)
						relations.append(x[:])
					if len(relations) > 0:
						relations = sorted(relations, key=lambda x: -x[-2])[:self.relation_limit]
						token.relation_embedding = torch.tensor(np.stack(relations))
					else:
						token.relation_embedding = torch.zeros(1, self.re_dim)
				if self.te_flag:
					ne_type = token.ne_type[:2].upper()
					token.type_embedding = torch.tensor(one_hot(self.t2i[ne_type] if ne_type in self.t2i else 0, self.te_dim))

	def postprocess(self, corpus: Corpus, entity_label: List) -> Corpus:
		"""
		corpus와 entity label을 받아서 corpus 내의 entity로 판명난 것들에 대해 entity명 및 타입 부여
		:param corpus: corpus
		:param entity_label: list of int
		:param make_copy: True일 경우 corpus를 deepcopy해서 복사본을 만듬. 원본이 필요한 경우 사용
		:return: entity가 표시된 corpus
		"""

		for entity, label in zip(corpus.entity_iter(), entity_label):
			target_entity = self.i2e[label] if label > 0 else ("_%d" % self.new_entity_count) + entity.surface.replace(" ", "_")
			entity.entity = target_entity
			self.new_entity_count += 1
		return corpus
