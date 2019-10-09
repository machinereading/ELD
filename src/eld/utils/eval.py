from sklearn.metrics import precision_recall_fscore_support, adjusted_rand_score

from . import DataModule
from ..utils import ELDArgs
from ...ds import CandDict
from ...utils import pickleload

class Evaluator:
	def __init__(self, args: ELDArgs, data: DataModule):
		self.ent_list = data.ent_list
		self.redirects = pickleload(args.redirects_path)
		self.surface_ent_dict = CandDict(self.ent_list, pickleload(args.entity_dict_path), self.redirects)
		self.out_kb_threhold = args.out_kb_threshold
		self.e2i = data.e2i
		self.oe2i = data.oe2i

	def evaluate(self, corpus, new_ent_pred, idx_pred, new_ent_label, idx_label):
		assert corpus.eld_len == len(new_ent_pred) == len(idx_pred) == len(new_ent_label) == len(idx_label)
		print(new_ent_label, new_ent_pred)
		print(idx_pred, idx_label)
		in_kb_prec, in_kb_rec, in_kb_f1, _ = precision_recall_fscore_support(new_ent_label, new_ent_pred, average="binary")
		# print(in_kb_prec, in_kb_rec, in_kb_f1)

		# give entity uri to each cluster
		pe_dark_id_to_ge_entity_map = {}

		pred_cluster = []
		gold_cluster = []

		for e, new_ent, idx in zip(corpus.eld_items, new_ent_pred, idx_pred):
			if not hasattr(e, "in_surface_dict"):
				e.in_surface_dict = e.surface in self.surface_ent_dict
			if new_ent:
				new_ent_gold_idx = self.oe2i[e.entity] + len(self.e2i) if e.entity in self.oe2i else self.e2i[e.entity]
				if idx not in pe_dark_id_to_ge_entity_map:
					pe_dark_id_to_ge_entity_map[idx] = {}
				if e.entity not in pe_dark_id_to_ge_entity_map[idx]:
					pe_dark_id_to_ge_entity_map[idx][new_ent_gold_idx] = 0
				pe_dark_id_to_ge_entity_map[idx][new_ent_gold_idx] += 1
				pred_cluster.append(idx)
				gold_cluster.append(new_ent_gold_idx)
		cluster_score_ari = adjusted_rand_score(gold_cluster, pred_cluster)
		# apply mapping
		mapping_result = {}
		for pred_idx, mapping in pe_dark_id_to_ge_entity_map.items():
			sorted_mapping = sorted(mapping.items(), key=lambda x: x[1], reverse=True)
			mapping_idx = -1
			for gold_idx, gold_count in sorted_mapping:
				if gold_idx >= len(self.e2i):
					mapping_idx = gold_idx
					break
			mapping_result[pred_idx] = mapping_idx
			for i in range(idx_pred.size(0)):
				if idx_pred[i] == pred_idx:
					idx_pred[i] = mapping_idx

		in_surface_dict_flags = [x.in_surface_dict for x in corpus.eld_items]
		new_ent_flags = [x.is_new_entity for x in corpus.eld_items]
		for f, l in zip(new_ent_flags, new_ent_label):
			assert (not f) == (not l)

		total_p, total_r, total_f1, total_support = precision_recall_fscore_support([x.item() for x in idx_label], [x.item() for x in idx_pred], average="micro")
		exist_p, exist_r, exist_f1, exist_support = precision_recall_fscore_support([x.item() for x, y in zip(idx_label, new_ent_flags) if not y], [x.item() for x, y in zip(idx_pred, new_ent_flags) if not y], average="micro")
		new_p, new_r, new_f1, new_support = precision_recall_fscore_support([x.item() for x, y in zip(idx_label, new_ent_flags) if y], [x.item() for x, y in zip(idx_pred, new_ent_flags) if y], average="micro")
		no_surface_p, no_surface_r, no_surface_f1, _ = precision_recall_fscore_support([x.item() for x, y in zip(idx_label, in_surface_dict_flags) if not y], [x.item() for x, y in zip(idx_pred, in_surface_dict_flags) if not y], average="micro")
		return (in_kb_prec, in_kb_rec, in_kb_f1), (total_p, total_r, total_f1), (exist_p, exist_r, exist_f1), (new_p, new_r, new_f1), (no_surface_p, no_surface_r, no_surface_f1), mapping_result
