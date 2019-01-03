def merge_item(j, result_list):
	if type(j) is not list:
		j = [j]

	result = {}
	target_name = ""
	target_json = None
	ind = 0

	for line in result_list[1:]:
		l = line.strip().split("\t")
		if len(l) == 1:
			target_name = l[0].split(" ")[0]
			# print(target_name)
			result[target_name] = []
			ind = 0
			for item in j:
				if item["fileName"] == target_name:
					target_json = item
					for item in target_json["entities"]:
						del item["candidates"]
						del item["keyword"]
						item["text"] = item["surface"]
						del item["surface"]

					break
			else:
				raise Exception("No such file name: %s" % target_name)
			continue
		target_json["entities"][ind]["entity"] = l[2]
		ind += 1
	return j


if __name__ == '__main__':
	print("merge result")
	import json
	with open("test_result_marking.txt", encoding="UTF8") as result_file, open("tta.json", encoding="UTF8") as j, open("tta_merged.json", "w", encoding="UTF8") as wf:
		jj = json.load(j)
		json.dump(merge_item(jj, result_file), wf, ensure_ascii=False, indent="\t")