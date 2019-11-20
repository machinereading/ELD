import sys

from src.el import EL
from src.utils import diriter, readfile, jsonload, jsondump, writefile
import csv
def split_to_sentence(j):
	sentences = j["text"].split("\n")
	stlens = list(map(len, sentences))
	for sentence_id, target_sentence in enumerate(sentences):
		sentence_start_idx = sum(stlens[:sentence_id]) + sentence_id if sentence_id > 0 else 0
		sent_ents = [x for x in j["entities"] if sentence_start_idx <= x["start"] < sentence_start_idx + len(target_sentence)]
		part = {
			"text"    : target_sentence,
			"entities": sent_ents,
			"fileName": j["fileName"] + "_%d" % sentence_id
		}
		for ent in part["entities"]:
			ent["start"] -= sentence_start_idx
			ent["end"] -= sentence_start_idx
			assert target_sentence[ent["start"]:ent["end"]] == ent["surface"]
		yield part

def overlap(s1, e1, s2, e2):
	return s1 <= s2 < e1 or s2 <= s1 < e2

def merge_el(original, elresult):
	if elresult == {}: return original
	for newentity in elresult["entities"]:
		for originalentity in original["entities"]:
			if overlap(originalentity["start"], originalentity["end"], newentity["start"], newentity["end"]):
				break
		else:
			original["entities"].append(newentity)
	original["entities"] = sorted(original["entities"], key=lambda x: x["start"])
	return original

def fix_json_keys(j):
	x = list(map(len, j["text"].split("\n")))
	y = [sum(x[:i]) for i in range(len(x))][1:]
	# print("----------------")
	except_item = []
	for e in j["entities"]:
		# if (e["datatype"] if "datatype" in e else e["dataType"]) == "namu": continue
		try:
			e["dataType"] = e["datatype"]
			del e["datatype"]
		except: pass
		try:
			e["surface"] = e["text"]
			del e["text"]
		except: pass
		start = e["start"]
		lsum = x[0]
		mod = 0
		for l in x[1:]:
			if start < lsum: break
			lsum += l
			mod += 1
		if j["text"][e["start"]:e["end"]] == e["surface"]: continue
		e["start"] += mod
		e["end"] += mod

		if j["text"][e["start"]:e["end"]] != e["surface"]:
			except_item.append(e)
	# if len(except_item) > len(y):
	# 	print(jname)
	# 	continue
	j["entities"] = [e for e in j["entities"] if e not in except_item]
	for e in j["entities"]:
		assert j["text"][e["start"]:e["end"]] == e["surface"]
	return j

def fix_entity_index(j):
	j["entities"] = sorted(j["entities"], key=lambda x: x["start"])
	entities = j["entities"]
	del_list = []
	for i, item in enumerate(entities):
		if item["dataType"] == "namu":
			for n in entities[i + 1:]:
				if n["dataType"] == "namu": continue
				if overlap(item["start"], item["end"], n["start"], n["end"]):
					del_list.append(n)
		else:
			for n in entities[i + 1:]:
				if overlap(item["start"], item["end"], n["start"], n["end"]):
					del_list.append(item)
					break

	j["entities"] = [x for x in entities if x not in del_list]
	for i, item in enumerate(j["entities"]):
		assert j["text"][item["start"]:item["end"]] == item["surface"], "entity text mismatch: %s / %s" % (j["text"][item["start"]:item["end"]], item["surface"])
		for item2 in j["entities"][i + 1:]:
			assert not overlap(item["start"], item["end"], item2["start"], item2["end"]), "entity overlap %d / %d" % (item["start"], item2["start"])
	return j

def el2re(j, targets):
	text = j["text"]
	lastidx = 0
	sentences = text.split("\n")
	stlens = list(map(len, sentences))
	j["entities"] = [x for x in j["entities"] if "entity" in x]
	for ent in [x for x in j["entities"] if x["entity"] in targets]:
		start = ent["start"]
		target_idx = 0
		lsum = stlens[0]
		for l in stlens[1:]:
			if start < lsum: break
			lsum += l + 1
			target_idx += 1
		target_sentence = sentences[target_idx]
		if len(target_sentence) > 512: continue
		sentence_start_idx = sum(stlens[:target_idx]) + target_idx if target_idx > 0 else 0
		if target_sentence[(ent["start"] - sentence_start_idx):(ent["end"] - sentence_start_idx)] != ent["surface"]:
			print("namu")
			print(ent["start"], ent["surface"], target_sentence, target_sentence[ent["start"] - sentence_start_idx:ent["end"] - sentence_start_idx])
			print(ent["start"] - sentence_start_idx, ent["end"] - sentence_start_idx)
			continue
		sentence_end_idx = sum(stlens[:target_idx + 1]) + target_idx
		target_entities = [e for e in j["entities"] if sentence_start_idx <= e["start"] < sentence_end_idx]
		pivot_index = target_entities.index(ent)
		for ent_idx, e in enumerate(target_entities):
			if e["start"] == ent["start"]: continue
			if target_sentence[(e["start"] - sentence_start_idx):(e["end"] - sentence_start_idx)] != e["surface"]:
				print("target")
				print(e["start"], e["surface"], target_sentence, target_sentence[(e["start"] - sentence_start_idx):(e["end"] - sentence_start_idx)])
				print(e["start"] - sentence_start_idx, e["end"] - sentence_start_idx)
				continue
			buf = []
			fs = min(e["start"] - sentence_start_idx, ent["start"] - sentence_start_idx)
			fe = min(e["end"] - sentence_start_idx, ent["end"] - sentence_start_idx)
			ss = max(e["start"] - sentence_start_idx, ent["start"] - sentence_start_idx)
			se = max(e["end"] - sentence_start_idx, ent["end"] - sentence_start_idx)
			assert fe <= ss, "%d / %d" % (fe + + sentence_start_idx + target_idx, ss + sentence_start_idx + target_idx)
			for x, y in [["e1", "e2"], ["e2", "e1"]]:
				buf.append(target_sentence[:fs])
				buf.append("<%s>" % x)
				buf.append(target_sentence[fs:fe])
				buf.append("</%s>" % x)
				buf.append(target_sentence[fe:ss])
				buf.append("<%s>" % y)
				buf.append(target_sentence[ss:se])
				buf.append("</%s>" % y)
				buf.append(target_sentence[se:])
				yield target_idx, ent_idx, pivot_index, "".join(buf)
				buf = []

def merge_re(result, re_input, js):
	lid = -1
	buf = []
	for i, (r1, r2) in enumerate(zip(result, re_input)):
		_, file_id, _, _, _ = r2
		file_id = int(file_id)
		if lid != -1 and lid != file_id:
			try:
				target_json = js[lid]
				sentences = target_json["text"].split("\n")
				stlens = list(map(len, sentences))
				for row1, row2 in buf:
					relation = row1[1]
					score = float(row1[-1])
					surface = [row1[0].split("<e%d>" % x)[1].split("</e%d>" % x)[0].replace('""', '"').replace("  ", " ") for x in [1, 2]]
					_, _, sentence_id, entity_id, pivot_id = row2
					sentence_id = int(sentence_id)

					target_sentence = sentences[int(sentence_id)]
					orig_sentence = row2[0]
					for s in ["<e1>", "</e1>", "<e2>", "</e2>"]:
						orig_sentence = orig_sentence.replace(s, "")
					assert target_sentence == orig_sentence
					# sentence_start_idx = target_json["text"].index(target_sentence)
					sentence_start_idx = sum(stlens[:sentence_id]) + sentence_id if sentence_id > 0 else 0
					sent_ents = [x for x in target_json["entities"] if sentence_start_idx <= x["start"] < sentence_start_idx + len(target_sentence)]
					target_entity = sent_ents[int(entity_id)]
					pivot_entity = sent_ents[int(pivot_id)]
					idx_diff = int(entity_id) - int(pivot_id)
					if target_entity["surface"].replace("  ", " ") not in surface:
						# print(lid, fn, target_entity["surface"], surface)
						continue
					# assert target_entity["surface"] in surface

					if pivot_entity["surface"].replace("  ", " ") not in surface:
						# print(lid, fn, pivot_entity["surface"], surface)
						continue
					# assert pivot_entity["surface"] in surface
					ent_idx = surface.index(pivot_entity["surface"].replace("  ", " "))
					if "relation" not in pivot_entity:
						pivot_entity["relation"] = []
					pivot_entity["relation"].append((idx_diff, relation, score, "incoming" if ent_idx == 1 else "outgoing"))
				yield target_json
				buf = []
			except Exception as e:
				import traceback
				traceback.print_exc()
				buf = []
		buf.append([r1, r2])
		lid = file_id
if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("mode", choices=["re_input", "re_merge", "entity_mapping"])
	parser.add_argument("input_file", type=str)
	parser.add_argument("output_file", type=str)
	parser.add_argument("--mapping_file", type=str)
	args = parser.parse_args()
	if args.mode == "re_input":
		NAMU_RAW_HOME = "/home/minho/namu/cpages/"
		target_entities = [x for x in readfile(args.input_file)]
		d = {}
		for item in target_entities:
			x = item.split("\t")
			if len(x) > 1:
				d[x[0]] = x[1]
			else:
				d[x[0]] = x[0]
		target_entities = d
		targets = []
		print("Filtering target entities")
		for j in map(jsonload, diriter(NAMU_RAW_HOME)):
			for part in split_to_sentence(j):
				target = False
				for entity in part["entities"]:
					if entity["entity"] in target_entities:
						entity["entity"] = target_entities[entity["entity"]]
						target = True
				if target: targets.append(part)



		print("Running EL")
		el = EL()
		el_result = el(*[x["text"] for x in targets])
		merged = []
		print("Merging EL result")
		for original, item in zip(targets, el_result):
			merged.append(merge_el(original, item))
		print("Fixing index")
		fixed = list(map(fix_json_keys, merged))
		fixed = list(map(fix_entity_index, fixed))
		print("Preparing RE")
		re_input = []
		for i, j in enumerate(fixed):
			for sentence_id, entity_id, pivot_id, txt in el2re(j, target_entities):
				re_input.append([txt, i, sentence_id, entity_id, pivot_id])
		with open("re_input.csv", "w", encoding="UTF8") as f:
			writer = csv.writer(f, delimiter=",")
			for item in re_input:
				writer.writerow(item)
		jsondump(fixed, "re_json_input.json")
	elif args.mode == "re_merge":
		with open("re_output.csv", encoding="UTF8") as f:
			reader = csv.reader(f, delimiter=",")
			re_result = [x for x in reader]
		with open("re_input.csv", encoding="UTF8") as f:
			reader = csv.reader(f, delimiter=",")
			re_input = [x for x in reader]
		fixed = jsonload("re_json_input.json")
		print("Merging RE result")
		print(len(re_result), len(re_input), len(fixed))
		re_merged = [j for j in merge_re(re_result, re_input, fixed)]

		jsondump(re_merged, args.output_file)
	else:
		mapping = jsonload(args.mapping_file)
		j = jsonload(args.input_file)
		for item in j:
			for ent in item["entities"]:
				if ent["entity"] in mapping:
					ent["entity"] = mapping[ent["entity"]]
		jsondump(j, args.output_file)
