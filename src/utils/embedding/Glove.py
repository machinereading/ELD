from .AbstractEmbedding import AbstractEmbedding

class Glove(AbstractEmbedding):
	def load_embedding(self, corpus_name):
		self.wv = {}
		with open(self.save_path+corpus_name, encoding="UTF8") as f:
			for line in f.readlines():
				x = line.split(" ")
				self.wv[x[0]] = x[1:]
		for v in self.wv.values():
			self.dim = len(v)
			break

	@property
	def file_prefix(self):
		return "glove"