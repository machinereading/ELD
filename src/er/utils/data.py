import re

import torch

from ...utils import jsonload, dictload, readfile
from ...utils.KoreanUtil import char_to_elem_ind
from ...ds import *
from typing import Dict

class NERDataset(torch.utils.data.Dataset):
	def __init__(self, data: Corpus, token_index_dict: Dict, tag_index_dict: Dict):
		self.data = data
		self.token2i = token_index_dict
		self.tag2i = tag_index_dict
		self.maxlen = max(map(len, data))

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		data = self.data[index]

		return torch.tensor([self.token2i[x] if x in self.token2i else 0 for x, _ in data] + ([0] * (self.maxlen - len(data)))), \
		       torch.tensor([self.tag2i[x] for _, x in data] + ([0] * (self.maxlen - len(data)))), \
		       len(data)

class DataModule:
	def __init__(self, mode, args):
		self.corpus = None
		self.w2i = [x for x in readfile(args.w2i)]
		self.jamo_limit = args.jamo_limit

	def initialize_tensor(self):
		longest_sent = 0
		longest_word = 0
		for sentence in self.corpus:
			longest_sent = max(longest_sent, len(sentence))
			for token in sentence:
				token.wi = dictload(self.w2i, token.surface)
				token.ji = []
				for char in token.surface:
					ci = char_to_elem_ind(char)
					token.ji += ci
					longest_word = max(longest_word, len(ci))



def cw2conll(cw_data):
	text = cw_data["text"]
	result = []
	ent = sorted(cw_data["entities"], key=lambda x: x["start"])
	for entity in ent:
		text = text[:entity["start"]] + "<" + text[entity["start"]:entity["end"]] + ">" + text[entity["end"]:]
		for entity in ent:
			entity["start"] += 2
			entity["end"] += 2

	text = re.sub(r"[^ ㄱ-ㅎㅏ-ㅣ가-힣a-z-A-Z0-9<>]+", "", text)
	sent = []
	token_flag = False
	for token in text.split(" "):
		tokens = token.split(">")
		l = len(tokens)
		# print(tokens)
		for i, token in enumerate(tokens):
			if len(token) == 0: continue
			if i != l - 1:
				token += ">"

			# print(token)
			token_buf = []
			if token[0] == "<":
				token_flag = True
				if ">" in token:
					token_flag = False
					t0, t1 = token.split(">")
					sent.append(" ".join([str(len(sent)), t0[1:], "_", "_", "S/E"]))
					if len(t1) > 0:
						sent.append(" ".join([str(len(sent)), t1, "_", "_", "O"]))
				else:
					sent.append(" ".join([str(len(sent)), token[1:], "_", "_", "B/E"]))
				continue
			if token_flag:
				if ">" in token:
					token_flag = False
					t0, t1 = token.split(">")
					sent.append(" ".join([str(len(sent)), t0, "_", "_", "E/E"]))
					if len(t1) > 0:
						sent.append(" ".join([str(len(sent)), t1, "_", "_", "O"]))
				else:
					sent.append(" ".join([str(len(sent)), token, "_", "_", "I/E"]))
			else:
				sent.append(" ".join([str(len(sent)), token, "_", "_", "O"]))
	return sent

def sentence2conll(sentence):
	return [" ".join([i, token.text, "_", "_", "O"]) for i, token in enumerate(sentence)]

if __name__ == '__main__':
	import os

	target_dir = "corpus/crowdsourcing_formatted/"
	with open("corpus/er/crowdsourcing1.conll", "w", encoding="UTF8") as f:
		for item in os.listdir(target_dir):
			f.write("\n".join(cw2conll(jsonload(target_dir + item))) + "\n\n")
