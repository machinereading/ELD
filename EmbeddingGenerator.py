import src.utils.embedding
from src.utils.datamodule.DataParser import CoNLLParser
from itertools import chain
cp = CoNLLParser("corpus/wiki_conll/")
j = src.utils.embedding.JamoEmbedding("wiki", 180, "word2vec")


def load(g):
	for m, _, _ in g:
		for char in m:
			yield char
# for item in chain(cp.get_trainset(), cp.get_devset()):
# 	print(item)
j.train_embedding("wiki", load(chain(cp.get_trainset(), cp.get_devset())), 5)
