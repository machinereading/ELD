def merge_item(j, result_dict, delete_candidate=True):
	if type(j) is not list:
		j = [j]

	target_name = ""
	# ind = 0

	for doc_name, pred in result_dict.items():
		# ind = 0
		# print("----")
		for item in j:
			if "fileName" not in item:
				continue
			if item["fileName"] == doc_name:
				target_json = item
				for item in target_json["entities"]:
					if delete_candidate:
						del item["candidates"]
						del item["answer"]
					# to get precise result, don't remove candidates
					item["text"] = item["surface"]
					item["dataType"] = item["ne_type"]
					del item["ne_type"]
					# del item["surface"]
				break
		else:
			raise Exception("No such file name: %s" % target_name)
		# print(l[2], target_json["entities"][ind]["keyword"])
		target_json["entities"] = sorted(target_json["entities"], key=lambda x: x["start"])
		ind = 0
		for m, g, p in pred:
			if p == "#UNK#":
				p = "NOT_IN_CANDIDATE"
			target_json["entities"][ind]["entity"] = p
			ind += 1

	# target_json["entities"][ind]["entity"] = l[2]
	# print(target_json["entities"][ind]["start"])
	# ind += 1
	return j

def merge_item_with_corpus(sentences, result_dict):
	for s, r in zip(sentences, result_dict):
		for entity in r["entities"]:
			target_voca = s.find_token_by_index(entity["start"])
			if target_voca is not None:
				target_voca.entity = entity["entity"] if "entity" in entity else "NOT_IN_CANDIDATE"
	return sentences

if __name__ == '__main__':
	print("merge result")
	import json

	with open("test_result_marking.txt", encoding="UTF8") as result_file, open("tta.json", encoding="UTF8") as j, open(
			"tta_merged.json", "w", encoding="UTF8") as wf:
		jj = json.load(j)
		json.dump(merge_item(jj, result_file), wf, ensure_ascii=False, indent="\t")
