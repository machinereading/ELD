from ...ds import *
from ...utils import readfile
from .args import ELDArgs
class DataModule:
	def __init__(self, args: ELDArgs): # TODO
		self.corpus = Corpus.load_corpus(args.corpus_dir)
		self.w2i = {w: i+1 for i, w in enumerate(readfile(args.word_file))}
		self.e2i = {w: i+1 for i, w in enumerate(readfile(args.entity_file))}
		self.r2i = {w: i+1 for i, w in enumerate(readfile(args.relation_file))}

	def generate_tensor(self):
		for sentence in self.corpus:
			for token in sentence:
				pass
