from .Vocabulary import Vocabulary
from ..utils import KoreanUtil
class Cluster():
	def __init__(self, target_entity):
		self.cluster = set([])
		self.target_entity = None
		self.id = -1
		self.max_jamo = 0

		self.has_tensor = False # tensor initialized
		self.has_pad_tensor = False # after padding
		
		self._jamo = []
		self._lctx_words = []
		self._rctx_words = []
		self._lctx_entities = []
		self._rctx_entities = []

	def add_elem(self, vocab):
		assert type(vocab) is Vocabulary
		# assert vocab not in self.cluster, vocab
		self.cluster.add(vocab)
		vocab.cluster = self
		je = []
		for char in vocab.surface:
			je += KoreanUtil.char_to_elem_ind(char)
		vocab.jamo_elem = je
		if len(je) > self.max_jamo:
			self.max_jamo = len(je)

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
		c = cls()
		c.id = json["id"]
		c.target_entity = json["target_entity"]
		for item in json["cluster"]:
			v = Vocabulary.from_json(item)
			assert v.cluster == c.id
			je = []
			for char in v.surface:
				je += KoreanUtil.char_to_elem_ind(char)
			v.jamo_elem = je
			c.cluster.add(v)
			v.cluster = c
		c.max_jamo = max([len(x.jamo_elem) for x in c])
		return c

	@property
	def vocab_tensors(self):
		if not self.has_tensor:
			for vocab in self:
				self._jamo.append(vocab.jamo_elem)
				self._lctx_words.append(vocab.lctxw_ind)
				self._rctx_words.append(vocab.rctxw_ind)
				self._lctx_entities.append(vocab.lctxe_ind)
				self._rctx_entities.append(vocab.rctxe_ind)
			self.has_tensor = True
			# add padding? no need to do in this level - padding must be performed in global level

		return self._jamo, self._lctx_words, self._rctx_words, self._lctx_entities, self._rctx_entities, 1 if not self.target_entity else 0

	def update_tensor(self, jamo, wlctx, wrctx, elctx, erctx):
		self._jamo = jamo
		self._lctx_words = wlctx
		self._rctx_words = wrctx
		self._lctx_entities = elctx
		self._rctx_entities = erctx
		self.has_pad_tensor = True