from src.ds import Corpus, Cluster

class LinkBasedEC:
	def __init__(self):
		pass

	def __call__(self, corpus):
		clusters = {}
		for sentence in corpus:
			for token in sentence:
				if token.entity not in clusters:
					clusters[token.entity] = Cluster(token.entity)
				clusters[token.entity].add_elem(token)
		corpus.cluster = clusters
		return corpus
