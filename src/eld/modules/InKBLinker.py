from abc import ABC, abstractmethod

import torch

from ..utils import ELDArgs
from ...ds import CandDict, Vocabulary
from ...el import EL
from ...utils import pickleload, readfile

class InKBLinker(ABC):
	@abstractmethod
	def __call__(self, *voca: Vocabulary):
		return None

	@abstractmethod
	def update_entity(self, surface, entity, emb):
		pass

class MulRel(InKBLinker):
	def __init__(self, args: ELDArgs, surface_ent_dict=None):
		self.el_module = EL(surface_ent_dict=surface_ent_dict)

	def __call__(self, *voca: Vocabulary):
		token_idx = [x.token_idx for x in voca]
		sentences = [x.parent_sentence for x in voca]
		result = self.el_module(*sentences)
		return [x.get_token_idx(idx).el_pred_entity for idx, x in zip(token_idx, result)]

	def update_entity(self, surface, entity, emb):
		self.el_module.data.surface_ent_dict.add_instance(surface, entity)

		self.el_module.ranker.model.entity_voca.add_instance(entity)
		if self.el_module.ranker.model.entity_voca.size() - self.el_module.ranker.model.entity_embedding.weight.size(0) == 1:
			self.el_module.ranker.model.entity_embedding.weight = torch.cat([self.el_module.ranker.model.entity_embedding.weight, emb])

		self.el_module.ranker.prerank_model.entity_voca.add_instance(entity)
		if self.el_module.ranker.prerank_model.entity_voca.size() - self.el_module.ranker.prerank_model.entity_embedding.weight.size(0) == 1:
			self.el_module.ranker.prerank_model.entity_embedding.weight = torch.cat([self.el_module.ranker.prerank_model.entity_embedding.weight, emb])

class PEM(InKBLinker):
	def __init__(self, args: ELDArgs, surface_ent_dict=None):
		self.ent_list = [x for x in readfile(args.ent_list_path)]
		self.redirects = pickleload(args.redirects_path)
		self.surface_ent_dict = CandDict(self.ent_list, pickleload(args.entity_dict_path), self.redirects) if surface_ent_dict is None else surface_ent_dict

	def __call__(self, *voca: Vocabulary):
		result = []
		for item in voca:
			res = self.surface_ent_dict[item.surface]
			# print(item.surface, res)
			result.append(res[0][0] if len(res) > 0 else "NOT_IN_CANDIDATE")
		return result

	def update_entity(self, surface, entity, emb):
		self.surface_ent_dict.add_instance(surface, entity)

class Dist(InKBLinker):

	def __init__(self, args: ELDArgs):
		pass

	def __call__(self, *voca: Vocabulary):
		pass
