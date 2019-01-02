from copy import deepcopy
from collections import namedtuple, defaultdict

from copy import deepcopy
from collections import namedtuple, defaultdict

import numpy as np
from ..utils import GlobalValues as gl
Entity = namedtuple("Entity", "e_type start_offset end_offset")

def collect_named_entities(tokens):
	"""
	Creates a list of Entity named-tuples, storing the entity type and the start and end
	offsets of the entity.
	:param tokens: a list of labels
	:return: a list of Entity named-tuples
	"""

	named_entities = []
	start_offset = None
	end_offset = None
	ent_type = None

	for offset, tag in enumerate(tokens):

		token_tag = tag

		if token_tag == 'O':
			if ent_type is not None and start_offset is not None:
				end_offset = offset - 1
				named_entities.append(Entity(ent_type, start_offset, end_offset))
				start_offset = None
				end_offset = None
				ent_type = None

		elif ent_type is None:
			ent_type = token_tag[2:]
			start_offset = offset

		elif ent_type != token_tag[2:]:
			end_offset = offset - 1
			named_entities.append(Entity(ent_type, offset - 1, end_offset))

			# start of a new entity
			ent_type = token_tag[2:]
			start_offset = offset
			end_offset = None

	# catches an entity that goes up until the last token
	if ent_type and start_offset and end_offset is None:
		named_entities.append(Entity(ent_type, start_offset, offset))

	return named_entities

def is_same_entity(a, b):
	return a["name"] == b["name"] and a["start_index"] == b["start_index"] and a["type"] == b["type"]

def compute_metrics(true_named_entities, pred_named_entities):

	eval_metrics = {'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurius': 0}
	target_tags_no_schema = []
	for label in gl.ner.label_dict.keys():
		l = label.split("-")
		if len(l) > 1:
			target_tags_no_schema.append(l[-1])

	# overall results
	evaluation = {'strict': deepcopy(eval_metrics), 'ent_type': deepcopy(eval_metrics), 'boundary': deepcopy(eval_metrics)}

	# results by entity type
	evaluation_agg_entities_type = {e: deepcopy(evaluation) for e in target_tags_no_schema}
	true_which_overlapped_with_pred = []  # keep track of entities that overlapped

	# go through each predicted named-entity
	for pred in pred_named_entities:
		found_overlap = False

		# check if there's an exact match, i.e.: boundary and entity type match
		flag = False
		for item in true_named_entities:
			if is_same_entity(pred, item):
				flag = True
				break
		if flag:
			true_which_overlapped_with_pred.append(pred)
			evaluation['strict']['correct'] += 1
			evaluation['ent_type']['correct'] += 1
			evaluation['boundary']['correct'] += 1

			# for the agg. by e_type results
			evaluation_agg_entities_type[pred["type"]]['strict']['correct'] += 1
			evaluation_agg_entities_type[pred["type"]]['ent_type']['correct'] += 1

		else:

			# check for overlaps with any of the true entities
			for true in true_named_entities:

				# check for an exact boundary match but with a different e_type
				if true["start_index"] <= pred["end_index"] and pred["start_index"] <= true["end_index"] and true["type"] != pred["type"]:

					# overall results
					evaluation['strict']['incorrect'] += 1
					evaluation['ent_type']['incorrect'] += 1
					evaluation['boundary']['correct'] += 1

					# aggregated by entity type results
					evaluation_agg_entities_type[pred["type"]]['strict']['incorrect'] += 1
					evaluation_agg_entities_type[pred["type"]]['ent_type']['incorrect'] += 1

					true_which_overlapped_with_pred.append(true)
					found_overlap = True
					break

				# check for an overlap (not exact boundary match) with true entities
				elif pred["start_index"] <= true["end_index"] and true["start_index"] <= pred["end_index"]:
					true_which_overlapped_with_pred.append(true)
					evaluation['boundary']['incorrect'] += 1
					if pred["type"] == true["type"]:  # overlaps with the same entity type
						# overall results
						evaluation['strict']['incorrect'] += 1
						evaluation['ent_type']['correct'] += 1

						# aggregated by entity type results
						evaluation_agg_entities_type[pred["type"]]['strict']['incorrect'] += 1
						evaluation_agg_entities_type[pred["type"]]['ent_type']['correct'] += 1

						found_overlap = True
						break

					else:  # overlaps with a different entity type
						# overall results
						evaluation['strict']['incorrect'] += 1
						evaluation['ent_type']['incorrect'] += 1

						# aggregated by entity type results
						evaluation_agg_entities_type[pred["type"]]['strict']['incorrect'] += 1
						evaluation_agg_entities_type[pred["type"]]['ent_type']['incorrect'] += 1

						found_overlap = True
						break

			# count spurius (i.e., over-generated) entities
			if not found_overlap:
				# overall results
				evaluation['strict']['spurius'] += 1
				evaluation['ent_type']['spurius'] += 1
				evaluation['boundary']['spurius'] += 1

				# aggregated by entity type results
				evaluation_agg_entities_type[pred["type"]]['strict']['spurius'] += 1
				evaluation_agg_entities_type[pred["type"]]['ent_type']['spurius'] += 1

	# count missed entities
	for true in true_named_entities:
		flag = False
		for item in true_which_overlapped_with_pred:
			if is_same_entity(true, item):
				flag = True
				break
		if flag:
			continue
		else:
			# overall results
			evaluation['strict']['missed'] += 1
			evaluation['ent_type']['missed'] += 1
			evaluation['boundary']['missed'] += 1

			# for the agg. by e_type
			evaluation_agg_entities_type[true["type"]]['strict']['missed'] += 1
			evaluation_agg_entities_type[true["type"]]['ent_type']['missed'] += 1

	# Compute 'possible', 'actual', according to SemEval-2013 Task 9.1
	for eval_type in ['strict', 'ent_type', 'boundary']:
		correct = evaluation[eval_type]['correct']
		incorrect = evaluation[eval_type]['incorrect']
		partial = evaluation[eval_type]['partial']
		missed = evaluation[eval_type]['missed']
		spurius = evaluation[eval_type]['spurius']

		# possible: nr. annotations in the gold-standard which contribute to the final score
		evaluation[eval_type]['possible'] = correct + incorrect + partial + missed

		# actual: number of annotations produced by the NER system
		evaluation[eval_type]['actual'] = correct + incorrect + partial + spurius

		actual = evaluation[eval_type]['actual']
		possible = evaluation[eval_type]['possible']

		if eval_type == 'partial_matching':
			precision = (correct + 0.5 * partial) / actual if actual > 0 else 0
			recall = (correct + 0.5 * partial) / possible if possible > 0 else 0
		else:
			precision = correct / actual if actual > 0 else 0
			recall = correct / possible if possible > 0 else 0

		evaluation[eval_type]['precision'] = precision
		evaluation[eval_type]['recall'] = recall
	return evaluation, evaluation_agg_entities_type

def metric_sum(a, b):
	a_eval, a_type = a
	b_eval, b_type = b
	for key in a_eval.keys():
		for kkey in a_eval[key].keys():
			a_eval[key][kkey] += b_eval[key][kkey]
	for t in a_type.keys():
		for key in a_type[t].keys():
			for kkey in a_type[t][key]:
				a_type[t][key][kkey] += b_type[t][key][kkey]

	return a_eval, a_type

def test_metric(s, a, l):
	tp_exact = tp_sub = 0
	fp_exact = fp_sub = 0
	fn_exact = fn_sub = 0
	
	a_word = list(map(lambda x: [x["name"], x["type"]], a))
	l_word = list(map(lambda x: [x["name"], x["type"]], l))
	correct_word = []
	sub_correct_word = []
	if len(a_word) == 0:
		return -1, -1
	elif len(a_word) > 0 and len(l_word) == 0:
		return [[[0,0,0],[0,0,0]] for _ in range(2)]
	
	for item in l_word:
		if item in a_word:
			tp_exact += 1
			correct_word.append(item)
		else:
			fp_exact += 1
			flag = False
			for aw in a_word:
				if (item[0] in aw[0] or aw[0] in item[0]) and item[1] == aw[1]:
					tp_sub += 1
					flag = True
					sub_correct_word.append(aw)
					break
			if not flag:
				fp_sub += 1
	for item in a_word:
		if item in correct_word: continue
		if item not in l_word:
			fn_exact += 1
			if item not in sub_correct_word:
				fn_sub += 1
	exact_precision = tp_exact / (tp_exact+fp_exact)
	exact_recall = tp_exact / (tp_exact+fn_exact)
	ef = 2*exact_precision*exact_recall/(exact_precision+exact_recall) if (exact_precision+exact_recall) > 0 else 0
	tp_sub += tp_exact
	sub_precision = tp_sub / (tp_sub+fp_sub)
	sub_recall = tp_sub / (tp_sub+fn_sub)
	sf = 2*sub_precision*sub_recall/(sub_precision+sub_recall) if (sub_precision+sub_recall) > 0 else 0
	typed = [(tp_exact, fp_exact, fn_exact), (tp_sub, fp_sub, fn_sub)]

	# without type matching
	tp_exact = tp_sub = 0
	fp_exact = fp_sub = 0
	fn_exact = fn_sub = 0
	correct_word = []
	sub_correct_word = []
	l_word = list(map(lambda x: x[0], l_word))
	a_word = list(map(lambda x: x[0], a_word))
	for item in l_word:
		if item in a_word:
			tp_exact += 1
			correct_word.append(item)
		else:
			fp_exact += 1
			flag = False
			for aw in a_word:
				if (item in aw or aw in item):
					tp_sub += 1
					flag = True
					sub_correct_word.append(aw)
					break
			if not flag:
				fp_sub += 1
	for item in a_word:
		if item in correct_word: continue
		if item not in l_word:
			fn_exact += 1
			if item not in sub_correct_word:
				fn_sub += 1
	exact_precision = tp_exact / (tp_exact+fp_exact)
	exact_recall = tp_exact / (tp_exact+fn_exact)
	ef = 2*exact_precision*exact_recall/(exact_precision+exact_recall) if (exact_precision+exact_recall) > 0 else 0
	tp_sub += tp_exact
	sub_precision = tp_sub / (tp_sub+fp_sub)
	sub_recall = tp_sub / (tp_sub+fn_sub)
	sf = 2*sub_precision*sub_recall/(sub_precision+sub_recall) if (sub_precision+sub_recall) > 0 else 0
	untyped = [(tp_exact, fp_exact, fn_exact), (tp_sub, fp_sub, fn_sub)]
	# postprocess_result = test_metric(s, a, postprocess(s, l, DEFAULT_POSTPROCESS_SEQUENCE))
	# print(typed, untyped, postprocess_result)
	return typed, untyped

def f1(tp, fp, fn):
	return prf(tp, fp, fn)[-1]

def prf(tp, fp, fn):
	prec = tp/(tp+fp) if tp+fp > 0 else 0
	rec = tp/(tp+fn) if tp+fn > 0 else 0
	return prec, rec, 2*(prec*rec)/(prec+rec) if prec > 0 and rec > 0 else 0

def dictadd(dict1, dict2):
	result = dict1
	for item in dict2:
		if item not in result:
			result[item] = dict2[item]
			continue
		for val in dict2[item]:
			for vval in dict2[item][val]:
				result[item][val][vval] += dict2[item][val][vval]
	return result