from typing import List

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from ...ds import Corpus
from ..utils import ELDArgs
from ...utils import TimeUtil

class ELDDataset(Dataset):
	def __init__(self, mode, corpus: Corpus, args: ELDArgs, filter_list=None, limit=None):
		if filter_list is None:
			filter_list = []
		self.mode = mode
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
		self.eld_items = []
		limit = 0 if limit is None or type(limit) is not int or limit < 0 else limit
		if len(filter_list) > 0:
			for token in self.corpus.eld_items:
				if token.entity in filter_list:
					self.eld_items.append(token)
					if 0 < limit <= len(self.eld_items):
						break
		else:
			self.eld_items = self.corpus.eld_items

	@TimeUtil.measure_time
	def __getitem__(self, index):
		# return ce, cl, lcwe, lcwl, rcwe, rcwl, lcee, lcel, rcee, rcel, re, rl, te, tl, lab
		# entity에 대한 주변 단어? -> we와 ee는 context, re는 다른 개체와의 관계, te는 자기 type
		def pad(tensor, pad_size):
			if tensor.dim() == 1:
				tensor = tensor.unsqueeze(0)
			add_size = pad_size - tensor.size(0)
			if add_size >= 0:
				return F.pad(tensor, [0, 0, 0, pad_size - tensor.size(0)])
			else:
				return tensor[:pad_size, :]

		# return torch.cat((tensor, torch.zeros(pad_size - tensor.size()[0], emb_dim, dtype=torch.float64)))
		target = self.eld_items[index]
		for item in target.tensor[:-3]:
			if item is None:
				print(target.tensor)
		ce, we, lwe, rwe, lee, ree, re, te, new_ent, ee_label, eidx = target.tensor
		cl = wl = lwl = rwl = lel = rel = rl = tl = 0
		if self.ce_flag:
			cl = ce.size()[0]
			ce = pad(ce, self.max_jamo_len_in_word)
		if self.we_flag:
			wl = we.size()[0]
			we = pad(we, self.max_word_len_in_entity)
		else:
			we = torch.zeros(1, dtype=torch.float)
		if self.wce_flag:
			if len(lwe) == 0:
				lwe = [torch.zeros(1, self.we_dim, dtype=torch.float)]
			if len(rwe) == 0:
				rwe = [torch.zeros(1, self.we_dim, dtype=torch.float)]
			lwe = torch.cat(lwe, dim=0).view(-1, self.we_dim)[-self.window_size:]
			rwe = torch.cat(rwe, dim=0).view(-1, self.we_dim)[:self.window_size]
			lwl = lwe.size()[0]
			rwl = rwe.size()[0]
			lwe = pad(lwe, self.window_size)
			rwe = pad(rwe, self.window_size)
		else:
			lwe = torch.zeros(1, dtype=torch.float)
			rwe = torch.zeros(1, dtype=torch.float)
		if self.ee_flag:
			if len(lee) == 0:
				lee = [torch.zeros(1, self.ee_dim, dtype=torch.float)]
			if len(ree) == 0:
				ree = [torch.zeros(1, self.ee_dim, dtype=torch.float)]
			lee = torch.cat(lee, dim=0).view(-1, self.ee_dim)[-self.window_size:]
			ree = torch.cat(ree, dim=0).view(-1, self.ee_dim)[:self.window_size]
			lel = lee.size()[0]
			rel = ree.size()[0]
			lee = pad(lee, self.window_size)
			ree = pad(ree, self.window_size)
		else:
			lee = torch.zeros(1, dtype=torch.float)
			ree = torch.zeros(1, dtype=torch.float)
		if self.re_flag:
			re = target.relation_embedding
			rl = re.size()[0]
			re = pad(re, self.r_limit)
		if self.te_flag:
			te = target.type_embedding
			tl = te.size()[0]
		else:
			te = torch.zeros(1, dtype=torch.float)
		if self.mode in ["train", "test"]:
			return ce, cl, we, wl, lwe, lwl, rwe, rwl, lee, lel, ree, rel, re, rl, te, tl, new_ent, ee_label, eidx, index
		else:
			return ce, cl, we, wl, lwe, lwl, rwe, rwl, lee, lel, ree, rel, re, rl, te, tl

	def __len__(self):
		return len(self.eld_items)
