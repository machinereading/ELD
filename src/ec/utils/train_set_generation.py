from ...utils import jsondump, readfile
import os
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
	el_result = []
	for item in el_module.predict(corpus, form="PLAIN_SENTENCE"):
		el_result += item
	# print(el_result)

	for item in el_result:
		for entity in item["entities"]:
			surface = entity["text"]
			ent = entity["entity"]
			if ent not in result:
				result[ent] = set([])
			result[ent].add(surface)
	jsondump(result, "data/ec/ds.json")


def generate_trainset(corpus_dir):
	if corpus_dir[-1] != "/": corpus_dir +="/"
	result = {}
	for item in os.listdir(corpus_dir):
		for line in readfile(corpus_dir+item):
			_, s, o, _, _ = line.split("\t")
			o = o.replace("kbox.kaist.ac.kr/resource/", "")
			if o not in result:
				result[o] = set([])
			result[o].add(s)

	with open("data/ec/train.txt", "w", encoding="UTF8") as f:
		ind = 0
		for k, v in result.items():
			f.write("c%d {" % ind)
			f.write(", ".join(["'"+x+"'" for x in v]))
			f.write("}\n")
			ind += 1

def trainset_statistics(train_set_file):
	if type(train_set_file) is str:
		train_set_file = readfile(train_set_file)
	else:
		train_set_file = train_set_file.readlines()
	clusters = 0
	lens = 0
	for line in train_set_file:
		data = " ".join(line.strip().split(" ")[1:])
		x = data.split(", ")
		clusters += 1
		lens += len(x)
	print(clusters, lens, lens / clusters)

if __name__ == '__main__':
	trainset_statistics("data/ec/train.txt")
	trainset_statistics("../ref/synsetmine/data/Wiki/train-cold.set")
	trainset_statistics("../ref/synsetmine/data/NYT/train-cold.set")
	trainset_statistics("../ref/synsetmine/data/PubMed/train-cold.set")
