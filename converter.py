import os
import json
source_dir = "corpus/crowdsourcing/"
target_dir = "corpus/crowdsourcing_processed/"
for item in os.listdir(source_dir):
	with open(source_dir+item, encoding="UTF8") as f, open(target_dir+item, "w", encoding="UTF8") as wf:
		j = json.load(f)
		result = {}
		result["original_text"] = j["text"]
		result["NE"] = []
		result["fileName"] = j["fileName"]
		for item in j["entities"]:
			result["NE"].append({
				"text": item["surface"],
				"keyword": item["keyword"],
				"char_start": item["start"],
				"char_end": item["end"],
				"type": ""
				})
		json.dump(result, wf, ensure_ascii=False, indent="\t")