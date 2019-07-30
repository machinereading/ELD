import operator
from sklearn.metrics.cluster import adjusted_rand_score

f1 = lambda p, r: 2 * p * r / (p + r) if p + r > 0 else 0

def evaluate(iter_el_result, label):
	"""

	:param iter_el_result: Corpus
	:param label: list of dict
	:return: analysis result
	"""
	# find entity in label
	found_entity = 0
	total_dark = 0
	el_count = {"TP": 0, "P": 0, "R": 0}
	additional_el_count = {"TP": 0, "P": 0, "R": 0}
	nae_count = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
	gold_only = {"TP": 0, "P": 0, "R": 0}
	nae = "NOT_AN_ENTITY"
	nic = "NOT_IN_CANDIDATE"
	id_to_entity_map = {}
	correct_cluster = []
	predicted_cluster = []
	cclk2i = {}
	pclk2i = {}
	sent_result = []
	cluster_assigned = 0
	for sentence, ls in zip(iter_el_result, label):
		for l in ls["entities"]:
			for e in sentence.entities:
				if e.char_ind == l["start"] and e.surface == l["surface"]:
					ne = e
					break
			else:
				continue

			try:
				# if new entity, then it should have cluster id
				cluster_id = int(ne.entity)
				if cluster_id not in id_to_entity_map:
					id_to_entity_map[cluster_id] = {}
				if l["entity"] not in id_to_entity_map[cluster_id]:
					id_to_entity_map[cluster_id][l["entity"]] = 0
				id_to_entity_map[cluster_id][l["entity"]] += 1
				cluster_assigned += 1
			except:
				# marked_entity = ne.entity
				# correct_entity = l["manual_entity"] if "manual_entity" in l else l["entity"]
				# if marked_entity != nae:
				# 	el_count["P"] += 1
				# 	if marked_entity == correct_entity:
				# 		el_count["TP"] += 1
				# if correct_entity != nae:
				# 	el_count["R"] += 1
				# if "dark_entity" in l and l["dark_entity"]:
				# 	total_dark += 1
				# 	if marked_entity != nic and "_fake" not in marked_entity:
				# 		found_entity += 1
				pass

	# register clusters with majority voting
	ide = {}
	for i, v in id_to_entity_map.items():
		maxk = max(v.items(), key=operator.itemgetter(1))[0]
		ide[i] = maxk

	for sentence, ls in zip(iter_el_result, label):
		for l in ls["entities"]:
			for e in sentence.entities:
				if e.char_ind == l["start"] and e.surface == l["surface"]:
					ne = e
					break
			else:
				continue
			correct_entity = l["manual_entity"] if "manual_entity" in l else l["entity"]
			l["answer"] = correct_entity
			tagged = "manual_entity" in l
			cluster_flag = False
			try:
				cluster_id = int(ne.entity)
				marked_entity = ide[cluster_id]
				cluster_flag = True
				if correct_entity != nae:
					pass
			except:
				marked_entity = ne.entity
			if marked_entity == nic:
				marked_entity = nae
			l["predict"] = marked_entity
			if marked_entity != nae:
				if tagged:
					gold_only["P"] += 1
				else:
					el_count["P"] += 1
				additional_el_count["P"] += 1
				if marked_entity == correct_entity:
					additional_el_count["TP"] += 1
					if tagged:
						gold_only["TP"] += 1
					else:
						el_count["TP"] += 1
			if correct_entity != nae:
				additional_el_count["R"] += 1
				if tagged:
					gold_only["R"] += 1
				else:
					el_count["R"] += 1
			# nae eval

			if correct_entity != nae and marked_entity != nae:
				nae_count["TP"] += 1
			elif correct_entity != nae and marked_entity == nae:
				nae_count["FN"] += 1
			elif correct_entity == nae and marked_entity != nae:
				nae_count["FP"] += 1
			else:
				nae_count["TN"] += 1
			# clustering gen
			if cluster_flag:
				if marked_entity not in predicted_cluster:
					pclk2i[marked_entity] = len(pclk2i)
				predicted_cluster.append(pclk2i[marked_entity])
				if correct_entity not in correct_cluster:
					cclk2i[correct_entity] = len(cclk2i)
				correct_cluster.append(cclk2i[correct_entity])
			if "dark_entity" in l and l["dark_entity"]:
				total_dark += 1
				if marked_entity != nic and "_fake" not in marked_entity:
					found_entity += 1
		sent_result.append(ls) # TODO
	p = lambda x: x["TP"] / x["P"] if x["P"] > 0 else 0
	r = lambda x: x["TP"] / x["R"] if x["R"] > 0 else 0
	elp = el_count["TP"] / el_count["P"] if el_count["P"] > 0 else 0
	elr = el_count["TP"] / el_count["R"] if el_count["R"] > 0 else 0
	elf1 = f1(elp, elr)
	aelp = additional_el_count["TP"] / additional_el_count["P"] if additional_el_count["P"] > 0 else 0
	aelr = additional_el_count["TP"] / additional_el_count["R"] if additional_el_count["R"] > 0 else 0
	aelf1 = f1(aelp, aelr)
	naep = nae_count["TP"] / (nae_count["TP"] + nae_count["FP"]) if (nae_count["TP"] + nae_count["FP"]) > 0 else 0
	naer = nae_count["TP"] / (nae_count["TP"] + nae_count["FN"]) if (nae_count["TP"] + nae_count["FN"]) > 0 else 0
	naef1 = f1(naep, naer)
	gp = p(gold_only)
	gr = r(gold_only)
	gf = f1(gp, gr)
	# for c, p in zip(correct_cluster[:20], predicted_cluster[:20]):
	# 	print(c, p)
	ari = adjusted_rand_score(correct_cluster, predicted_cluster)
	print(el_count)
	print("EL Score: %.4f, %.4f, %.4f" % (elp * 100, elr * 100, elf1 * 100))
	print(additional_el_count)
	print("Additional EL Score: %.4f, %.4f, %.4f" % (aelp * 100, aelr * 100, aelf1 * 100))
	print("NAE Score: %.4f, %.4f, %.4f" % (naep * 100, naer * 100, naef1 * 100))
	print("ARI: %.4f" % ari)
	print("Gold only: %.4f, %.4f, %.4f" % (gp * 100, gr * 100, gf *100))
	print("Dark entity found: %d / %d" % (found_entity, total_dark))
	print("Cluster assigned: %d" % cluster_assigned)
	return {"EL": [elp, elr, elf1], "AEL": [aelp, aelr, aelf1], "NAE": [naep, naer, naef1], "ARI": ari, "DEFound": [found_entity, total_dark], "Gold": [gp, gr, gf]}, sent_result, ide
