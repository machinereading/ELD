from sklearn.metrics import precision_recall_fscore_support, adjusted_rand_score

from . import DataModule
from ..utils import ELDArgs
from ...ds import CandDict
from ...utils import pickleload, TimeUtil

class Evaluator:
	def __init__(self, args: ELDArgs, data: DataModule):
		self.ent_list = data.ent_list
		self.redirects = pickleload(args.redirects_path)
		self.surface_ent_dict = CandDict(self.ent_list, pickleload(args.entity_dict_path), self.redirects)
		self.out_kb_threhold = args.out_kb_threshold
		self.e2i = data.e2i
		self.oe2i = data.oe2i

	@TimeUtil.measure_time
	def evaluate(self, corpus, new_ent_pred, idx_pred, new_ent_label, idx_label):

		def record(target_dict, p, l):
			target_dict["Total"] += 1
			target_dict["R"] += 1
			if p == l:
				target_dict["Correct"] += 1
			if p != 0:
				target_dict["P"] += 1
				if p == l:
					target_dict["TP"] += 1


		assert corpus.eld_len == len(new_ent_pred) == len(idx_pred) == len(new_ent_label) == len(idx_label)
		kb_expect_prec, kb_expect_rec, kb_expect_f1, _ = precision_recall_fscore_support(new_ent_label, new_ent_pred, average="binary")

		# give entity uri to each cluster
		pe_dark_id_to_ge_entity_map = {}

		pred_cluster = []
		gold_cluster = []
		total, in_kb, out_kb, no_surface = [{"TP": 0, "P": 0, "R": 0, "Total": 0, "Correct": 0} for _ in range(4)]
		kb_pred = {"Total": 0, "Correct": 0}
		for e, new_ent, idx in zip(corpus.eld_items, new_ent_pred, idx_pred):
			idx = idx.item()
			new_ent = new_ent.item()
			if not hasattr(e, "in_surface_dict"):
				e.in_surface_dict = e.surface in self.surface_ent_dict
			if new_ent:
				new_ent_gold_idx = self.oe2i[e.entity] + len(self.e2i) if e.entity in self.oe2i else self.e2i[e.entity]
				if idx not in pe_dark_id_to_ge_entity_map:
					pe_dark_id_to_ge_entity_map[idx] = {}
				if new_ent_gold_idx not in pe_dark_id_to_ge_entity_map[idx]:
					pe_dark_id_to_ge_entity_map[idx][new_ent_gold_idx] = 0
				pe_dark_id_to_ge_entity_map[idx][new_ent_gold_idx] += 1
				pred_cluster.append(idx)
				gold_cluster.append(new_ent_gold_idx)
		cluster_score_ari = adjusted_rand_score(gold_cluster, pred_cluster)
		# apply mapping
		mapping_result = {}
		# print(pe_dark_id_to_ge_entity_map)
		for pred_idx, mapping in pe_dark_id_to_ge_entity_map.items(): # out-kb pred to label matching
			"""
			등록 순서에 따라 index를 받기 때문에 index 변환 과정이 필요함
			"""
			sorted_mapping = sorted(mapping.items(), key=lambda x: x[1], reverse=True)
			for gold_idx, gold_count in sorted_mapping:
				if gold_idx >= len(self.e2i): # out-kb index만 매핑 수행해야 함. out-kb로 판단했는데 in-kb로 매핑한다는 것은 애초에 틀린 것.
					mapping_idx = gold_idx
					break
			else: # out-kb 매핑 불가 - 틀림
				mapping_idx = 0 # not in candidate(wrong)
			mapping_result[pred_idx] = mapping_idx
			for i in range(idx_pred.size(0)):
				if idx_pred[i].item() == pred_idx:
					idx_pred[i] = mapping_idx

		in_surface_dict_flags = [x.in_surface_dict for x in corpus.eld_items]
		new_ent_flags = [x.is_new_entity for x in corpus.eld_items]
		for e, nep, ip, nel, il, in_surface_dict in zip(corpus.eld_items, new_ent_pred, idx_pred, new_ent_label, idx_label, in_surface_dict_flags):
			# set record targets
			record_targets = [total]
			if nel:
				record_targets.append(out_kb)
			else:
				record_targets.append(in_kb)
			if not in_surface_dict:
				record_targets.append(no_surface)
			# kb expectation score
			kb_pred["Total"] += 1
			if nep == nel:
				kb_pred["Correct"] += 1
			# linking score
			for x in record_targets:
				record(x, ip, il)


		for f, l in zip(new_ent_flags, new_ent_label):
			assert (not f) == (not l) # tensor value가 항상 true로 찍힐 위험이 있기 때문에 not을 넣어서 해결..?
		p = lambda d: d["TP"] / d["P"]
		r = lambda d: d["TP"] / d["R"]
		f1 = lambda p, r: (2 * p * r / (p + r) if p+r > 0 else 0)
		total_p = p(total)
		total_r = r(total)
		total_f1 = f1(total_p, total_r)
		# total_acc = acc(total)

		in_kb_p = p(in_kb)
		in_kb_r = r(in_kb)
		in_kb_f1 = f1(in_kb_p, in_kb_r)
		# in_kb_acc = acc(in_kb)

		out_kb_p = p(out_kb)
		out_kb_r = r(out_kb)
		out_kb_f1 = f1(out_kb_p, out_kb_r)
		# out_kb_acc = acc(out_kb)

		no_surface_p = p(no_surface)
		no_surface_r = r(no_surface)
		no_surface_f1 = f1(no_surface_p, no_surface_r)
		# no_surface_acc = acc(no_surface)
		# total_p, total_r, total_f1, total_support = precision_recall_fscore_support([x.item() for x in idx_label], [x.item() for x in idx_pred], average="micro")
		# exist_p, exist_r, exist_f1, exist_support = precision_recall_fscore_support([x.item() for x, y in zip(idx_label, new_ent_flags) if not y], [x.item() for x, y in zip(idx_pred, new_ent_flags) if not y], average="micro")
		# new_p, new_r, new_f1, new_support = precision_recall_fscore_support([x.item() for x, y in zip(idx_label, new_ent_flags) if y], [x.item() for x, y in zip(idx_pred, new_ent_flags) if y], average="micro")
		# no_surface_p, no_surface_r, no_surface_f1, _ = precision_recall_fscore_support([x.item() for x, y in zip(idx_label, in_surface_dict_flags) if not y], [x.item() for x, y in zip(idx_pred, in_surface_dict_flags) if not y],
		#                                                                                average="micro")
		return (kb_expect_prec, kb_expect_rec, kb_expect_f1), (total_p, total_r, total_f1), (in_kb_p, in_kb_r, in_kb_f1), (out_kb_p, out_kb_r, out_kb_f1), (no_surface_p, no_surface_r, no_surface_f1), cluster_score_ari, mapping_result
