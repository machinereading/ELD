###########################
# label-related functions #
###########################

import numpy as np
import re
LABELS = 21
DEFAULT_POSTPROCESS_SEQUENCE = ["find_same_surface", "remove_punctuation", "eojeol_process", "mark_time", "merge_entities"]

def attention(char):
	if ord("가") <= ord(char) <= ord("힣"): return True
	if ord("0") <= ord(char) <= ord("9"): return True
	if ord("a") <= ord(char) <= ord("z"): return True
	if ord("A") <= ord(char) <= ord("Z"): return True
	return False
def label_int_to_str(item):
	if type(item) is str: return item
	if item == 0:
		return "O"
	plom = "PLOMT"
	bilu = "BILU"
	item -= 1
	return (bilu[item % 4]+"/"+plom[item // 4])
def label_no(label):
	bilu = "BILU"
	plom = "PLOMT"
	if label == "O": return 0
	x = label.split("/")
	return bilu.index(x[0])+plom.index(x[1])*4+1

def one_hot(ind, max):
	result = [0] * max
	result[ind-1] = 1
	return result


def one_hot_to_label(l):
	return l.index(1)

def encode_label(tag, mark_type=True):
	# BILU*5 + O
	if tag == "O": return 0
	bilu = "BILU".index(tag[0])
	if not mark_type:
		return bilu
	ty = tag[-1]
	if ty not in "PLOMT": return None
	return "PLOMT".index(ty)*4+bilu+1


def decode_label(sentence, label, score=None):
	if type(sentence) is SeparatedSentence:
		sentence = sentence.original_sentence
	entities = []
	temp = ""
	pm = 0
	if score is None:
		score = [None] * len(label)
	label = list(map(label_int_to_str, label))
	ind = 0
	sin = 0
	sc = 0
	for c, l, s in zip(sentence, label, score):
		if l[0] == "U":
			entities.append({"name": sentence[ind], "type": l[-1], "start_index": ind, "end_index": ind+1, "score": float(sc / (ind-sin+1)) if s is not None else 0})
			ind += 1
			continue
		if l[0] == "B":
			pm += 1
			sin = ind
			temp += c
			if s is not None:
				sc += s[label_no(l)]
		if pm == 1 and l[0] == "I":
			temp += c
			if s is not None:
				sc += s[label_no(l)]
		if pm == 1 and l[0] == "L":
			temp += c
			if s is not None:
				sc += s[label_no(l)]
			entities.append({"name": temp, "type": l[-1], "start_index": sin, "end_index": ind+1, "score": float(sc / (ind-sin+1))}) # surface, type, start index, end index, score
			temp = ""
			pm = 0
		ind += 1
	entities.sort(key=lambda x: x["start_index"])
	i = 1
	for item in entities:
		item["index"] = i
		i += 1
	return entities

def decode_morph_labels(sentence, morphs, labels):
	morph_index = 0
	char_index = 0
	last_label = ""
	result = []
	while True:
		if char_index >= len(sentence): 
			# print(result)
			# return decode_label(sentence, result)
			return result
		char = sentence[char_index]
		next_morph = morphs[morph_index]
		next_label = label_int_to_str(labels[morph_index])
		last_label = result[-1] if len(result) > 0 else ""
		if char == " ":
			char_index += 1
			morph_index += 1
			if last_label[0] in "B":
				result.append("I"+last_label[1:])
			elif last_label[0] in "LU":
				result.append("O")
			else:
				result.append(last_label)
		elif char in next_morph:
			char_index += 1
			label = next_label[0]

			if label != "O":
				label_type = next_label[-1]
			else:
				result.append("O")
				continue
			if label == "B":
				if char == next_morph[0]:
					result.append("B/%s" % label_type)
				else:
					result.append("I/%s" % label_type)
			elif label == "I":
				b_flag = False
				result.append("I/%s" % label_type)
			elif label == "L":
				if char == next_morph[-1]:
					result.append("L/%s" % label_type)
				else:
					result.append("I/%s" % label_type)
			elif label == "U":
				if len(next_morph) == 1:
					result.append("U/%s" % label_type)
					continue
				if char == next_morph[0]:
					result.append("B/%s" % label_type)
				elif char == next_morph[-1]:
					result.append("L/%s" % label_type)
				else:
					result.append("I/%s" % label_type)
			
		else:
			morph_index += 1



# 왠갖 postprocessing method

def postprocess(sentence, entities, sequence=DEFAULT_POSTPROCESS_SEQUENCE):
	def is_inside_entity(start_index, end_index, entities):
		result = []
		for item in entities:
			if item["start_index"] <= start_index < item["end_index"]:
				result.append(item)
			elif item["start_index"] <= end_index < item["end_index"]:
				result.append(item)
		return result
	def find_same_surface(sentence, entities):
		result = entities
		entities = list(filter(lambda x: len(x["name"]) >= 3, entities)) # 길이 2 이하인 entity는 오류일 수 있음. 거른다!
		entities = sorted(list(entities), key=lambda x: -len(x["name"]))
		remove_entity = []
		eb = list(map(lambda x: (x["start_index"], x["end_index"]), entities))
		for entity, tag in list(map(lambda x: (x["name"], x["type"]), entities)):
			added = True
			not_entity_index = []
			while added:
				entity_index = []
				nsi = 0
				added = False
				while True:
					try:
						nsi = sentence.index(entity, nsi)
						entity_index.append(nsi)
						nsi += len(entity)
					except Exception:
						break
				for nsi in entity_index:
					new_entity = True
					for si, ei in eb:
						# print(si, ei, nsi, nsi+len(entity))
						if nsi <= si and ei <= nsi+len(entity) and not (nsi == si and nsi+len(entity) == ei) and (si, ei) not in remove_entity:
							remove_entity.append((si, ei))
						elif si <= nsi < ei or si <= nsi+len(entity) < ei:
							new_entity = False
							continue
						
					if new_entity:
						flag = False
						original_entity = entity
						# new_entity = {"name": entity, "type": tag, "start_index": nsi, "end_index": nsi+len(entity), "score": 0}
						result.append({"name": entity, "type": tag, "start_index": nsi, "end_index": nsi+len(entity), "score": 0})
						eb = list(map(lambda x: (x["start_index"], x["end_index"]), entities+result))
						added = True
						break
		r = []

		for si, ei in remove_entity:
			for item in entities:
				if si == item["start_index"] and ei == item["end_index"]:
					r.append(item)
					break
		# if len(r) > 0 : print("R:", list(map(lambda x: (x["name"], x["start_index"]), r)))
		return list(filter(lambda x: x not in r, result))

	# entity가 어절 중간에서 시작한다면 어절의 맨 앞부터 포함하도록 변경		
	def eojeol_process(sentence, entities):
		
		remove_entities = []
		new_entities = []
		for entity in entities:
			new_entity = entity
			flag = False
			while new_entity["start_index"] > 0 and attention(sentence[new_entity["start_index"]-1]):
				# 어절 처음부터 entity가 되게 등록
				# print(sentence[nsi-1])
				for item in entities:
					if new_entity["start_index"] - 1 == item["end_index"]: # 다른 entity와 겹치게 되는 경우
						break
				new_entity["start_index"] -= 1
				# print(sentence[new_entity["start_index"]], new_entity["name"])
				new_entity["name"] = sentence[new_entity["start_index"]]+new_entity["name"]

				flag = True

			if flag:
				remove_entities.append(entity)
				new_entities.append(new_entity)

		entities = list(filter(lambda x: x not in remove_entities, entities))
		for item in new_entities:
			entities.append(item)
		return entities
	# 괄호같은거 다 지움
	def remove_punctuation(sentence, entities):

		return entities

	# 인접한 entity는 하나로 merge
	def merge_entities(sentence, entities):
		lastentity = None
		remove_entity = []
		added = True
		while added:
			added = False
			for entity in entities:
				if lastentity is not None and \
				   entity not in remove_entity and \
				   lastentity["type"][0] == "T" and \
				   entity["type"][0] == "T" and \
				   entity["start_index"] - lastentity["end_index"] == 1 and \
				   sentence[lastentity["end_index"]] == " ":
					lastentity["name"] += " %s" % entity["name"]
					lastentity["end_index"] = entity["end_index"]
					remove_entity.append(entity)
					added = True
				lastentity = entity
			entities = sorted(list(filter(lambda x: x not in remove_entity, entities)), key=lambda x: x["start_index"])

		return entities

	# re기반 datetime 체크
	def mark_time(sentence, entities):
		year_pattern = r"(기원전 )?[0-9]{2,}년(대?( 초| 말))?"
		month_pattern = r"(음력 )?[0-9]+월( [0-9]+일)?"
		for pattern in [year_pattern, month_pattern]:
			for item in re.finditer(pattern, sentence):
				start = item.start()
				end = item.end()
				if not len(is_inside_entity(start, end, entities)):
					entities.append({"name": sentence[start:end], "type": "T", "start_index": start, "end_index": end, "score":1.0})
		return entities

	def pairing(sentence, entities):
		start = "《"
		end = "》"
		i = 0
		sin, ein = 0, 0
		pairs = []
		for char in sentence:
			if char == start:
				sin = i+1
			if char == end:
				ein = i
				pairs.append((sin, ein))
			i+=1
		removes = []
		for sin, ein in pairs:
			dup = is_inside_entity(sin, ein, entities)
			if len(dup) > 0:
				for item in dup: removes.append(dup)
			entities.append({"name": sentence[sin:ein], "type": "M", "start_index": sin, "end_index": ein, "score":1.0})


	postprocess_methods = {
		"find_same_surface": find_same_surface, 
		"eojeol_process": eojeol_process, 
		"remove_punctuation": remove_punctuation, 
		"merge_entities": merge_entities, 
		"mark_time": mark_time
	}
	if type(sentence) is SeparatedSentence:
		sentence = sentence.original_sentence
	result = entities[:]
	guide_flag = False
	# print("original")
	# print(list(map(lambda x: (x["name"], x["start_index"]), result)))

	for item in sequence:
		if item in postprocess_methods:
			old_result = result
			result = sorted(postprocess_methods[item](sentence, result), key=lambda x: x["start_index"])
			# if result != old_result:
			# 	print("\t",item)
			# 	print("\t", list(map(lambda x: (x["name"], x["start_index"]), result)))
		else:
			print("%s is not in postprocess methods." % item)
			guide_flag = True
	if guide_flag:
		print("Please select one of below:\n%s" % ("\n".join([k for k, _ in postprocess_methods.items()])))
	# if result != entities:
	# 	print(entities)
	# 	print(result)
	return result


def wrap_entities_in_text(sent, entities, mark_type=False):
	s = "start_index"
	e = "end_index"
	t = "type"
	print(sent)
	print(entities)
	ent = sorted(entities, key=lambda x: x[s])
	se_pairs = [[x[s], x[e], x[t]] for x in ent]
	add = 3 if mark_type else 1
	for item in se_pairs:
		sent = sent[:item[0]]+"["+sent[item[0]:]
		for item1 in se_pairs:
			item1[0] += 1
			item1[1] += 1
		end = ":%s]" % item[2] if mark_type else "]"
		sent = sent[:item[1]]+(end)+sent[item[1]:]
		for item1 in se_pairs:
			item1[0] += add
			item1[1] += add
	return sent




# score: total score sequence
def calculate_crf_score(score, seq):
	pass




if __name__ == '__main__':
	x = SeparatedSentence("[[[체첸 공화국|체첸_공화국]]]<<AdministrativeRegion|LOC>> , 또는 줄여서 체첸(, , )은 [[[러시아의 공화국|러시아의_공화국]]]<<Country|LOC>> 이다.")
	print(x.sentences)
	# import json
	# import DataPrepareModule as dp
	# with open("test", encoding="UTF8") as f:
	# 	sentence, labels = next(dp.wikipedia_golden(f))
	# item = {}
	# x = decode_label(sentence, labels)
	# item["sentence"] = sentence
	# item["entities"] = x
	# # with open("eval.json", encoding="UTF8") as f:
	# # 	j = json.load(f)
	# # for item in j:
	# sentence = item["sentence"]

	# old_entities = item["entities"]
	# print(wrap_entities_in_text(sentence, old_entities))
	# print(old_entities)
	# new_entities = postprocess(sentence, old_entities, ["find_same_surface", "remove_punctuation", "eojeol_process"])
	# print(wrap_entities_in_text(sentence, new_entities))