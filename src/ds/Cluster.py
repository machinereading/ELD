from .Vocabulary import Vocabulary
class Cluster():
	def __init__(self):
		self.cluster = set([])
		self.target_entity = None
		self.id = -1
		self.vocab_tensors = {
			"lctx_words": [],
			"rctx_words": [],
			"lctx_entities": [],
			"rctx_entities": [],
			"error_type": []
		}
		# 가짜 entity marking은 어떻게? - vocab마다 체크함
		

	def add_elem(self, vocab):
		assert type(vocab) is Vocabulary
		# assert vocab not in self.cluster, vocab
		self.cluster.add(vocab)
		vocab.cluster = self

	def __iter__(self):
		for token in self.cluster:
			yield token

	def __len__(self):
		return len(self.cluster)

	def to_json(self):
		return {
			"id": self.id,
			"target_entity": self.target_entity,
			"cluster": [x.to_json() for x in self.cluster]
		}

	@classmethod
	def from_json(cls, json):
		c = Cluster()
		c.id = json["id"]
		for item in json["cluster"]:
			v = Vocabulary.from_json(item)
			assert v.cluster == c.id
			c.cluster.add(v)
			v.cluster = c
	