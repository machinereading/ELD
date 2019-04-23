def merge_item(j, result_dict, delete_candidate=True):
	if type(j) is not list:
		j = [j]

	result = {}
	target_name = ""
	target_json = None
	# ind = 0

	for doc_name, pred in result_dict.items():
		# ind = 0
		# print("----")
		for item in j:
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
					del item["surface"]
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


if __name__ == '__main__':
	print("merge result")
	import json
	with open("test_result_marking.txt", encoding="UTF8") as result_file, open("tta.json", encoding="UTF8") as j, open("tta_merged.json", "w", encoding="UTF8") as wf:
		jj = json.load(j)
		json.dump(merge_item(jj, result_file), wf, ensure_ascii=False, indent="\t")