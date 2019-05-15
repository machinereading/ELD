import torch
from torch.utils import data

from .args import ECArgs
from ...ds import *
from ...utils import readfile, KoreanUtil

class CorefDataset(data.Dataset):
	def __init__(self, corpus):
		self.corpus = corpus

	def __getitem__(self, index):
		target_sentence = self.corpus.sentences[index]
		return target_sentence.ec_word_tensor, target_sentence.ec_inds, target_sentence.ec_cluster_label

	def __len__(self):
		return len(self.corpus)

class DataModule:
	def __init__(self, args: ECArgs):
		self.args = args
		self.w2i = {w: i for i, w in enumerate(readfile(args.embedding_path + ".word"))}
		self.w_pad = len(self.w2i) + 1

	def prepare(self, input_data):
		# set form to corpus
		corpus = input_data
		if type(input_data) is list:
			corpus = Corpus()
			for data in input_data:
				sentence = Sentence.from_cw_form(data)
				corpus.add_sentence(sentence)

		buf = []
		indbuf = []
		clusterbuf = []
		for sentence in corpus:
			words = []
			inds = []
			cluster = []
			ind = 0
			for token in sentence:
				if token.is_entity:
					tokens = KoreanUtil.tokenize(token.surface)
					add = [self.w2i[x] if x in self.w2i else len(self.w2i) for x in tokens]
					words += add
					inds.append((ind, ind + len(add)))
					ind += len(add)
					cluster.append(token.ec_cluster_id)
				else:
					words.append(self.w2i[token.surface] if token.surface in self.w2i else len(self.w2i))
					ind += 1
			buf.append(words)
			indbuf.append(inds)
			clusterbuf.append(cluster)
		maxlen = max([len(x) for x in buf])
		buf = [x + [self.w_pad] * (maxlen - len(x)) for x in buf]
		for b, s in zip(buf, corpus):
			s.ec_word_tensor = torch.LongTensor(b)

		maxlen = max([len(x) for x in indbuf])
		indbuf = [x + [(-1, -1)] * (maxlen - len(x)) for x in indbuf]
		for b, s in zip(indbuf, corpus):
			s.ec_inds = torch.LongTensor(b)

		maxlen = max([len(x) for x in clusterbuf])
		clusterbuf = [x + [-1] * (maxlen - len(x)) for x in clusterbuf]
		for b, s in zip(clusterbuf, corpus):
			s.ec_cluster_label = torch.LongTensor(b)

		return input_data

	def decode(self, sentence, target_entity_id, ancestor_entity_id):
		pass
