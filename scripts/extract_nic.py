import json, os
from src.utils import jsonload, jsondump

target_dir = ["corpus/crowdsourcing_formatted/", "corpus/mta2_postprocessed/"]
result = []
for d in target_dir:
	for item in os.listdir(d):
		j = jsonload(d+item)
		for ent in j["entities"]:
			if ent["entity"] == "NOT_IN_CANDIDATE":
				context_entities = []
				for ent2 in j["entities"]:
					if ent != ent2 and abs(ent["start"] - ent2["start"]) < 30 and ent2["entity"] not in ["NOT_IN_CANDIDATE", "NOT_AN_ENTITY"]:
						context_entities.append(ent2)
				result.append({
					"surface": ent["surface"],
					"context": j["text"][max(0, ent["start"]-20):min(len(j["text"]), ent["end"]+20)],
					"neighbors": [x["entity"] for x in context_entities]
					})

jsondump(result, "corpus/dark_entity_candidates.json")