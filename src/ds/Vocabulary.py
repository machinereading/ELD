from typing import List

import torch

from . import Relation
from ..utils.KoreanUtil import decompose_sent


class Vocabulary:
	def __init__(self, surface, parent_sentence, token_ind=0, char_ind=0):
		self.surface: str = surface
		self.surface_idx: int = -1
		self.is_entity = False
		self.ne_type = None
		self.entity = None
		self.entity_idx = -1
		self.entity_in_kb = False
		self.char_ind = char_ind
		self.token_ind = token_ind
		self.parent_sentence = parent_sentence

		# reserved for EL
		self.el_pred_entity = None

		# reserved for EV
		self.cluster = -1
		self.error_type = -1  # -1: normal, 0: ER, 1: EL, 2: EC

		# reserved for ELD
		self.target = False
		self.char_embedding: torch.Tensor = torch.zeros(1, dtype=torch.float)
		self.word_embedding: torch.Tensor = torch.zeros(1, dtype=torch.float)  # embedding of self.surface
		self.entity_embedding: torch.Tensor = torch.zeros(1, dtype=torch.float)  # embedding of self.entity
		self.relation_embedding: torch.Tensor = torch.zeros(1, dtype=torch.float)
		self.type_embedding: torch.Tensor = torch.zeros(1, dtype=torch.float)
		self.eld_tensor_initialized: bool = False
		self.entity_label_embedding: torch.Tensor = torch.zeros(1, dtype=torch.float)
		self.entity_label_idx: int = -1
		self.relation: List[Relation] = []
		self.eld_pred_entity = None

		# some properties that will be initialized later
		self.lctxw_ind = None
		self.rctxw_ind = None
		self.lctxe_ind = None
		self.rctxe_ind = None
		self.jamo_ind = None
		self.tagged = False

	def __str__(self):
		return self.surface

	def __len__(self):
		return len(self.surface)

	def __repr__(self):
		return "%s: %s" % (self.surface, self.entity if self.entity is not None else "")

	def __eq__(self, other):
		if type(other) is not Vocabulary: return False
		return self.parent_sentence == other.parent_sentence and self.char_ind == other.char_ind

	def __hash__(self):
		return hash((self.parent_sentence, self.char_ind))

	@property
	def lctx(self):
		return self.parent_sentence.tokens[:self.token_ind]

	@property
	def rctx(self):
		return self.parent_sentence.tokens[self.token_ind + 1:]

	@property
	def lctx_ent(self):
		return [x for x in self.lctx if x.is_entity]

	@property
	def rctx_ent(self):
		return [x for x in self.rctx if x.is_entity]

	def to_json(self):
		return {
			"surface"        : self.surface,
			"entity"         : self.entity,
			"is_entity"      : self.is_entity,
			"entity_in_kb"   : self.entity_in_kb,
			"parent_sentence": self.parent_sentence.id,
			"cluster"        : self.cluster if type(self.cluster) is int else self.cluster.id,
			"error_type"     : self.error_type,
			"token_ind"      : self.token_ind,
			"char_ind"       : self.char_ind
		}

	@classmethod
	def from_json(cls, json):
		voca = cls(json["surface"], None, json["token_ind"], json["char_ind"])
		for k, v in json.items():
			setattr(voca, k, v)
		if not voca.is_entity:
			voca.error_type = 0
		if voca.is_entity and voca.entity_in_kb:
			voca.error_type = 1
		return voca

	# reserved for eld
	@property
	def tensor(self):
		"""
		tensor input of eld module
		:return: character embedding, word embedding, left word embedding, right word embedding, left entity embedding, right entity embedding, relation embedding, type embedding
		"""

		return self.char_embedding, \
		       self.word_embedding, \
		       [x.word_embedding for x in self.lctx], \
		       [x.word_embedding for x in self.rctx[::-1]], \
		       [x.entity_embedding for x in self.lctx_ent], \
		       [x.entity_embedding for x in self.rctx_ent[::-1]], \
		       self.relation_embedding, \
		       self.type_embedding, \
		       self.is_new_entity, \
		       self.entity_label_embedding,\
			   self.entity_label_idx

	@property
	def jamo(self):
		return decompose_sent(self.surface)

	@property
	def is_new_entity(self):
		return self.entity.startswith("namu_")

	def get_relative_token(self, relative_idx):
		return self.parent_sentence.entities[self.entity_idx + relative_idx]
