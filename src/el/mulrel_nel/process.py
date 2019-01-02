from . import etri2 as etri
from . import merge_result as merge_result
import os
from datetime import datetime
from .dataset_changer import change_to_conll, change_to_tsv
from .main import test
start_iter = 0
def process_el():
	
	cs_form = etri.change_etri_into_crowdsourcing_form()
	
	batch_size = 10000
	
	end = False
	it = 0
	logfile = open("error_log.txt", "w", encoding="UTF8")
	while not end:
		batch = []
		print(datetime.now(), "Start change into json")
		start_id = None
		# make batch
		while True:
			try:
				elem = next(cs_form)
				id = int(elem["fileName"])
				if start_id is None:
					start_id = id
					print(start_id)
				batch.append(elem)
				if id >= start_id + batch_size - 1:
					break
			except StopIteration:
				end = True
				break
		it += 1
		print(datetime.now(), "Done")
		if it < start_iter: continue
		print("Iteration %d: Start writing conll and tsv" % it)
		cs = []
		cands = []
		for item in batch:
			try:
				assert len(item["entities"]) > 0
				cs += change_to_conll(item).split("\n")
				cs.append("")
				cands += change_to_tsv(item)
			except Exception as e:
				logfile.write("\t".join(["Error:", item["fileName"], str(e)])+"\n")
		print(datetime.now(), "Done")
		print("Iteration %d: Start EL" % it)
		try:
			result = test(cs, cands)
		except Exception:
			print("Error on EL: ")
			continue
		merged_result = merge_result.merge_item_with_list(batch, result)
		print(datetime.now(), "Done")
		print("Iteration %d: Start writing result file" % it)
		kbox_prefix = "kbox.kaist.ac.kr/resource/"
		with open("el_result/result_%d.tsv" % it, "w", encoding="UTF8") as finalf:
			for sentence in merged_result:
				global_sid = sentence["fileName"]
				for entity in sentence["entities"]:
					try:
						if entity["entity"] not in ["#UNK#", "NIL"]:
							finalf.write("\t".join([global_sid, entity["text"], kbox_prefix+entity["entity"], str(entity["start"]), str(entity["end"])])+"\n")
					except:
						# print(entity)
						continue
		print(datetime.now(), "Done")

if __name__ == '__main__':

	process_el()