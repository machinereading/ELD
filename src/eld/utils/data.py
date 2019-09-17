import numpy as np
from torch.utils.data import Dataset

from src.el.utils.data import CandDict
from .args import ELDArgs
from ...ds import *
from ...utils import readfile, pickleload
from ...el.utils import CandidateDict

class ELDDataset(Dataset):
	def __init__(self, corpus: Corpus, args: ELDArgs):
		self.corpus = corpus
		self.max_token_len = max(map(len, self.corpus))

	def __getitem__(self, index):
		pass

	def __len__(self):
		pass

class DataModule:
	def __init__(self, args: ELDArgs, corpus_dir):  # TODO
		self.corpus = Corpus.load_corpus(corpus_dir)
		# index 0: not in dictionary
		self.ce_flag = args.use_character_embedding
		self.we_flag = args.use_word_context_embedding
		self.ee_flag = args.use_entity_context_embedding
		self.re_flag = args.use_relation_embedding
		self.te_flag = args.use_type_embedding
		self.ent_list = [x for x in readfile(args.ent_list_path)]
		self.redirects = pickleload(args.redirects_path)
		self.surface_ent_dict = CandDict(self.ent_list, pickleload(args.entity_dict_path), self.redirects)
		if self.ce_flag:
			self.c2i = {w: i + 1 for i, w in enumerate(readfile(args.character_file))}
			ce = np.load(args.character_embedding_file)
			self.character_embedding = np.stack([np.zeros(ce.shape[-1]), *ce])
			args.ce_dim = ce.shape[-1]
		if self.we_flag:
			self.w2i = {w: i + 1 for i, w in enumerate(readfile(args.word_file))}
			we = np.load(args.word_embedding_file)
			self.word_embedding = np.stack([np.zeros(we.shape[-1]), *we])
			args.we_dim = we.shape[-1]
		if self.ee_flag:
			self.e2i = {w: i + 1 for i, w in enumerate(readfile(args.entity_file))}
			ee = np.load(args.entity_embedding_file)
			self.entity_embedding = np.stack([np.zeros(ee.shape[-1]), *ee])
			args.ee_dim = ee.shape[-1]
		if self.re_flag:
			self.r2i = {w: i + 1 for i, w in enumerate(readfile(args.relation_file))}
			re = np.load(args.relation_embedding_file)
			self.relation_embedding = np.stack([np.zeros(re.shape[-1]), *re])
			args.re_dim = re.shape[-1]
		if self.te_flag:
			self.t2i = {w: i + 1 for i, w in enumerate(readfile(args.type_file))}
			te = np.load(args.type_embedding_file)
			self.type_embedding = np.stack([np.zeros(te.shape[-1]), *te])
			args.te_dim = te.shape[-1]

	def generate_tensor(self):
		for sentence in self.corpus:
			for token in sentence:
				if self.ce_flag:
					token.char_embedding = None
				if self.we_flag:
					token.word_embedding = self.word_embedding[self.w2i[token.surface]] if token.surface in self.w2i else np.zeros(100, np.float)
				if self.ee_flag:
					token.entity_embedding = self.entity_embedding[token.entity] if token.is_entity and token.entity in self.e2i else np.zeros(100, np.float)
				if self.re_flag:
					token.relation_embedding = None
				if self.te_flag:
					token.type_embedding = None

