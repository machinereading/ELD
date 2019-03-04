from ...ds.Token import Sentence
def crowdsourcing2sent(cw_dict):
	sentence = Sentence(cw_dict["text"])
	for entity in cw_dict["entities"]:
		sentence.add_ne(entity["start"], entity["end"], entity=entity["entity"])
	return sentence



if __name__ == '__main__':
	from .. import jsonload
	j = jsonload("corpus/el_golden_postprocessed_marked/100044.json")
	crowdsourcing2sent(j)