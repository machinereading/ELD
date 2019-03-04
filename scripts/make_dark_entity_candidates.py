from src.utils import jsonload, jsondump

import os
import re
target_dir = ["corpus/crowdsourcing_formatted/", "corpus/mta2_postprocessed/"]
result = []
for d in target_dir:
	for f in os.listdir(d):
		j = jsonload(d+f)
		file_name = j["fileName"]
		
		for entity in j["entities"]:
			if re.sub(r"([0-9]+년)? ?([0-9]+월)? ?([0-9]+일)?", "", entity["surface"]) == "": continue
			if entity["entity"] == "NOT_IN_CANDIDATE":
				result.append({
					"surface": entity["surface"],
					"neighbors": [x["entity"] for x in j["entities"] if x["entity"] not in ["NOT_IN_CANDIDATE", "NOT_AN_ENTITY"] and abs(x["start"] - entity["start"]) < 30],
					"context": j["text"][max(0, entity["start"] - 30):min(len(j["text"]), entity["end"]+30)],
					"fileName": file_name
					})

jsondump(result, "corpus/de_candidates.json")