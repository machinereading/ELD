from typing import List

import torch
from sklearn.metrics import precision_recall_fscore_support, adjusted_rand_score

from src.ds import Vocabulary
from . import DataModule
from ..utils import ELDArgs
from ...ds import CandDict
from ...utils import pickleload, TimeUtil

class Evaluator:
	def __init__(self, args: ELDArgs, data: DataModule):
		self.ent_list = data.ent_list
		self.redirects = pickleload(args.redirects_path)
		self.new_ent_threshold = args.new_ent_threshold
		self.surface_ent_dict = CandDict(self.ent_list, pickleload(args.entity_dict_path), self.redirects) # need original canddict
		self.e2i = data.e2i
		if hasattr(data, "oe2i"):
			self.oe2i = data.oe2i

	@TimeUtil.measure_time
	def evaluate(self, eld_items, new_ent_pred, idx_pred, new_ent_label, idx_label):
		def record(target_dict, p, l):
			target_dict["Total"] += 1
			target_dict["R"] += 1
			if p == l:
				target_dict["Correct"] += 1
			if p != 0:
				target_dict["P"] += 1
				if p == l:
					target_dict["TP"] += 1

		assert len(eld_items) == len(new_ent_pred) == len(idx_pred) == len(new_ent_label) == len(idx_label)
		kb_expect_prec, kb_expect_rec, kb_expect_f1, _ = precision_recall_fscore_support(new_ent_label, [1 if x > self.new_ent_threshold else 0 for x in new_ent_pred], average="binary")

		# give entity uri to each cluster
		pe_dark_id_to_ge_entity_map = {}  # {pred_idx: {gold_idx: count}}

		pred_cluster = []
		gold_cluster = []
		total_c, in_kb_c, out_kb_c, no_surface_c = [{"TP": 0, "P": 0, "R": 0, "Total": 0, "Correct": 0} for _ in range(4)]
		total_u, in_kb_u, out_kb_u, no_surface_u = [{"TP": 0, "P": 0, "R": 0, "Total": 0, "Correct": 0} for _ in range(4)]
		# kb_pred = {"Total": 0, "Correct": 0}
		for e, new_ent, idx in zip(eld_items, new_ent_pred, idx_pred):
			if type(idx) is torch.Tensor:
				idx = idx.item()
			if type(new_ent) is torch.Tensor:
				new_ent = new_ent.item()
			new_ent = new_ent > self.new_ent_threshold
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
		mapping_result_clustered = {}
		mapped_entity_clustered = {}  # 한번 assign된 entity의 re-assign 방지
		# {gold_idx: {pred_idx:count}}

		for pred_idx, mapping in pe_dark_id_to_ge_entity_map.items():
			for gold_idx, gold_count in mapping.items():
				if gold_idx not in mapped_entity_clustered:
					mapped_entity_clustered[gold_idx] = {}
				mapped_entity_clustered[gold_idx][pred_idx] = gold_count

		for k, v in mapped_entity_clustered.items():
			mapped_entity_clustered[k] = list(sorted([[k, v] for k, v in v.items()], key=lambda x: x[1], reverse=True))
		mapped_entity_clustered = [[k, v] for k, v in mapped_entity_clustered.items()]
		# 최대 cluster부터 linking 시작
		while len(mapped_entity_clustered) > 0:
			mapped_entity_clustered = list(sorted([[k, v] for k, v in mapped_entity_clustered], key=lambda x: max([y[1] for y in x[1]]), reverse=True))
			gold_idx, sorted_pred_index = mapped_entity_clustered.pop(0)
			pred_idx, _ = sorted_pred_index[0]
			# print(gold_idx, sorted_pred_index, pred_idx)
			mapping_result_clustered[pred_idx] = gold_idx if gold_idx >= len(self.e2i) else 0
			for item in mapped_entity_clustered:
				item[1] = list(filter(lambda x: x[0] != pred_idx, item[1]))
			mapped_entity_clustered = [x for x in mapped_entity_clustered if len(x[1]) > 0]

		for pred_idx, gc in pe_dark_id_to_ge_entity_map.items():
			if pred_idx not in mapping_result_clustered:
				mapping_result_clustered[pred_idx] = -list(sorted([[k, v] for k, v in gc.items()], key=lambda x: x[1], reverse=True))[0][0]
		# print(mapping_result_clustered)
		idx_pred_clustered = idx_pred[:]
		for pred_idx, mapping_idx in mapping_result_clustered.items():
			for i in range(len(idx_pred_clustered)):
				if idx_pred_clustered[i] == pred_idx:
					idx_pred_clustered[i] = mapping_idx
		# unclustered mapping
		mapping_result_unclustered = {}
		for pred_idx, mapping in pe_dark_id_to_ge_entity_map.items():  # out-kb pred to label matching
			# 등록 순서에 따라 index를 받기 때문에 index 변환 과정이 필요함
			sorted_mapping = sorted(mapping.items(), key=lambda x: x[1], reverse=True)
			for gold_idx, gold_count in sorted_mapping:
				if gold_idx >= len(self.e2i):  # out-kb index만 매핑 수행해야 함. out-kb로 판단했는데 in-kb로 매핑한다는 것은 애초에 틀린 것.
					mapping_idx = gold_idx
					break
			else:  # out-kb 매핑 불가 - 틀림
				mapping_idx = 0  # not in candidate(wrong)
			mapping_result_unclustered[pred_idx] = mapping_idx
		idx_pred_unclustered = idx_pred[:]

		for pred_idx, mapping_idx in mapping_result_unclustered.items():
			for i in range(len(idx_pred_unclustered)):
				if idx_pred_unclustered[i] == pred_idx:
					idx_pred_unclustered[i] = mapping_idx



		in_surface_dict_flags = [x.in_surface_dict for x in eld_items]
		new_ent_flags = [x.is_new_entity for x in eld_items]
		for e, nep, ipc, ipu, nel, il, in_surface_dict in zip(eld_items, new_ent_pred, idx_pred_clustered, idx_pred_unclustered, new_ent_label, idx_label, in_surface_dict_flags):
			# set record targets
			record_targets_c = [total_c]
			record_targets_u = [total_u]
			if nel:
				record_targets_c.append(out_kb_c)
				record_targets_u.append(out_kb_u)
			else:
				record_targets_c.append(in_kb_c)
				record_targets_u.append(in_kb_u)
			if not in_surface_dict:
				record_targets_c.append(no_surface_c)
				record_targets_u.append(no_surface_u)
			# kb expectation score
			# kb_pred["Total"] += 1
			# if (nep > self.new_ent_threshold) == (nel == 1):
			# 	kb_pred["Correct"] += 1
			# linking score
			for x in record_targets_c:
				record(x, ipc, il)
			for x in record_targets_u:
				record(x, ipu, il)

		# for f, l in zip(new_ent_flags, new_ent_label):
		# 	assert (not f) == (not l)  # tensor value와 int 비교를 위해 not 넣음
		p = lambda d: d["TP"] / d["P"] if d["P"] > 0 else 0
		r = lambda d: d["TP"] / d["R"] if d["R"] > 0 else 0
		f1 = lambda p, r: (2 * p * r / (p + r) if p + r > 0 else 0)
		total_c_p = p(total_c)
		total_c_r = r(total_c)
		total_c_f1 = f1(total_c_p, total_c_r)

		in_kb_c_p = p(in_kb_c)
		in_kb_c_r = r(in_kb_c)
		in_kb_c_f1 = f1(in_kb_c_p, in_kb_c_r)

		out_kb_c_p = p(out_kb_c)
		out_kb_c_r = r(out_kb_c)
		out_kb_c_f1 = f1(out_kb_c_p, out_kb_c_r)

		no_surface_c_p = p(no_surface_c)
		no_surface_c_r = r(no_surface_c)
		no_surface_c_f1 = f1(no_surface_c_p, no_surface_c_r)

		total_u_p = p(total_u)
		total_u_r = r(total_u)
		total_u_f1 = f1(total_u_p, total_u_r)

		in_kb_u_p = p(in_kb_u)
		in_kb_u_r = r(in_kb_u)
		in_kb_u_f1 = f1(in_kb_u_p, in_kb_u_r)

		out_kb_u_p = p(out_kb_u)
		out_kb_u_r = r(out_kb_u)
		out_kb_u_f1 = f1(out_kb_u_p, out_kb_u_r)

		no_surface_u_p = p(no_surface_u)
		no_surface_u_r = r(no_surface_u)
		no_surface_u_f1 = f1(no_surface_u_p, no_surface_u_r)
		# no_surface_acc = acc(no_surface)
		# total_p, total_r, total_f1, total_support = precision_recall_fscore_support([x.item() for x in idx_label], [x.item() for x in idx_pred], average="micro")
		# exist_p, exist_r, exist_f1, exist_support = precision_recall_fscore_support([x.item() for x, y in zip(idx_label, new_ent_flags) if not y], [x.item() for x, y in zip(idx_pred, new_ent_flags) if not y], average="micro")
		# new_p, new_r, new_f1, new_support = precision_recall_fscore_support([x.item() for x, y in zip(idx_label, new_ent_flags) if y], [x.item() for x, y in zip(idx_pred, new_ent_flags) if y], average="micro")
		# no_surface_p, no_surface_r, no_surface_f1, _ = precision_recall_fscore_support([x.item() for x, y in zip(idx_label, in_surface_dict_flags) if not y], [x.item() for x, y in zip(idx_pred, in_surface_dict_flags) if not y],
		#                                                                                average="micro")
		return (kb_expect_prec, kb_expect_rec, kb_expect_f1), \
		       ((total_c_p, total_c_r, total_c_f1), (total_u_p, total_u_r, total_u_f1)), \
		       ((in_kb_c_p, in_kb_c_r, in_kb_c_f1), (in_kb_u_p, in_kb_u_r, in_kb_u_f1)), \
		       ((out_kb_c_p, out_kb_c_r, out_kb_c_f1), (out_kb_u_p, out_kb_u_r, out_kb_u_f1)), \
		       ((no_surface_c_p, no_surface_c_r, no_surface_c_f1), (no_surface_u_p, no_surface_u_r, no_surface_u_f1)), \
		       cluster_score_ari, \
		       mapping_result_clustered, \
		       mapping_result_unclustered

	def evaluate_by_form(self, pred: List[str], gold: List[str]):
		def record(target_dict, p, l):
			target_dict["Total"] += 1
			target_dict["R"] += 1
			if p == l:
				target_dict["Correct"] += 1
			if p != "NOT_IN_CANDIDATE":
				target_dict["P"] += 1
				if p == l:
					target_dict["TP"] += 1
		assert len(pred) == len(gold)
		total, in_kb, out_kb= [{"TP": 0, "P": 0, "R": 0, "Total": 0, "Correct": 0} for _ in range(3)]
		kb_expectation = []
		kb_gold = []
		for p, g in zip(pred, gold):
			target = [total]
			newent_flag = g not in self.e2i
			if newent_flag:
				target.append(out_kb)
			else:
				target.append(in_kb)
			kb_gold.append(newent_flag)
			kb_expectation.append(p not in self.e2i)
			for item in target:
				record(item, p, g)

		p = lambda d: d["TP"] / d["P"] if d["P"] > 0 else 0
		r = lambda d: d["TP"] / d["R"] if d["R"] > 0 else 0
		f1 = lambda p, r: (2 * p * r / (p + r) if p + r > 0 else 0)
		discovery_p, discovery_r ,discovery_f, _ = precision_recall_fscore_support(kb_gold, kb_expectation, average="binary")
		total_p = p(total)
		total_r = r(total)
		total_f1 = f1(total_p, total_r)

		in_kb_p = p(in_kb)
		in_kb_r = r(in_kb)
		in_kb_f1 = f1(in_kb_p, in_kb_r)

		out_kb_p = p(out_kb)
		out_kb_r = r(out_kb)
		out_kb_f1 = f1(out_kb_p, out_kb_r)

		return (discovery_p, discovery_r ,discovery_f), (total_p, total_r, total_f1), (in_kb_p, in_kb_r, in_kb_f1), (out_kb_p, out_kb_r, out_kb_f1)
