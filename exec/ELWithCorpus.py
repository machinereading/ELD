from src.ds import Corpus
from src.el import EL

el = EL()
corpus = Corpus.load_corpus("result.json")

for item in el(*corpus):
	print(item)
