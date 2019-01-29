from ...utils import jsondump

def generate_clusters_from_dataset(d):
	result = {}
	for item in d:
		pass
	return result


def distant_supervision(corpus, el_module):
	"""
	Generate Distant supervision data from corpus using EL Module
	:param corpus: Corpus data. string list which contains one document in one element
	:type corpus: list
	:param el_module: EL Module
	:type el_module: EL
	"""
	result = {}
	el_result = el_module.predict(corpus, form="PLAIN_SENTENCE")
	# print(el_result)
	for item in el_result:
		for entity in item["entities"]:
			surface = entity["text"]
			ent = entity["entity"]
			if ent not in result:
				result[ent] = set([])
			result[ent].add(surface)
	jsondump(result, "data/ec/ds.json")