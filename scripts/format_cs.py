from src.utils import jsonload, jsondump

import os

t1 = "corpus/crowdsourcing_processed/"
t2 = "corpus/crowdsourcing_formatted/"
for f in os.listdir(t1):
	item = jsonload(t1+f)
	r = {
		"text": item["original_text"],
		"fileName": f,
		"entities": []
	}
	for ent in item["NE"]:
		r["entities"].append({
			"surface": ent["text"],
			"entity": ent["keyword"],
			"start": ent["char_start"],
			"end": ent["char_end"],
			"ne_type": ent["type"]
			})
	jsondump(r, t2+f)
