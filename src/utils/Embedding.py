import numpy as np
from gensim.models import KeyedVectors
def load_embedding(embedding_path, embedding_type):
	"""
	Input:
		embedding_path: str, path to embedding file
		embedding_type: one of word2vec, fasttext, glove
	Output:
		w2i: word-index dictionary
		embedding: np array
	"""
	if embedding_type not in ["word2vec", "fasttext", "glove"]: raise ValueError("Must be one of word2vec, fasttext, or glove")
	if embedding_type == "word2vec" or embedding_type == "fasttext":
		vec = KeyedVectors.load(embedding_path)
		w2i = {w: i for i, w in enumerate(vec.index2word)}
		arr = []
		for word in vec.index2word:
			arr.append(vec.wv[word])
		return w2i, np.array(arr)

	elif embedding_type == "glove":
		d = []
		e = []
		with open(embedding_path, encoding="UTF8") as f:
			for line in f.readlines():
				x = line.split(" ")
				d.append(x[0])
				e.append([float(v) for v in x[1:]])
		return {w:i for i, w in enumerate(d)}, np.array(e)