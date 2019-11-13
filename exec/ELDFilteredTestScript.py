import sys

from src.eld import VectorBasedELD
from src.utils import readfile, jsonload, writefile

filter_entity = [x for x in readfile("various_form_entities")]
files = []
j = jsonload("corpus/namu_entity_filenames.json")
for item in filter_entity:
	files += j[item]
model_name = sys.argv[1]

eld = VectorBasedELD("pred", model_name)
corpus = [jsonload(x) for x in files]
processed = []
for item in corpus:
	sentences = item["text"].split("\n")
	stlens = list(map(len, sentences))
	for sentence_id, target_sentence in enumerate(sentences):
		sentence_start_idx = sum(stlens[:sentence_id]) + sentence_id if sentence_id > 0 else 0
		sent_ents = [x for x in item["entities"] if sentence_start_idx <= x["start"] < sentence_start_idx + len(target_sentence)]
		part = {
			"text"    : target_sentence,
			"entities": sent_ents,
			"fileName": item["fileName"] + "_%d" % sentence_id
		}
		for ent in part["entities"]:
			ent["start"] -= sentence_start_idx
			ent["end"] -= sentence_start_idx
			assert target_sentence[ent["start"]:ent["end"]] == ent["surface"]
		processed.append(part)
writefile(eld(*processed), "ambiguous_result.tsv", processor=lambda x:"\t".join([str(y) for y in x]))


