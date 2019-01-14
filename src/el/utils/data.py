import json
import os
import pickle
from konlpy.tag import Okt
import socket
from functools import reduce
from .CandidateDict import CandidateDict
from ...utils import TimeUtil
import os

okt = Okt()
dbpedia_prefix = "ko.dbpedia.org/resource/"



# with open("data/el/unk_entity_calc.pickle", "rb") as f:
# 	ent_dict = pickle.load(f)
# with open("data/el/wiki_entity_cooccur.pickle", "rb") as f:
# 	ent_form = pickle.load(f)
# ent_form = ent_form.keys()
# with open("data/el/redirects.pickle", "rb") as f:
# 	redirects = pickle.load(f)
with open("data/el/wiki_entity_dict.json", encoding="UTF8") as f:
	candidate_dict = CandidateDict.from_file(f)
with open("data/el/kb_entities", encoding="UTF8") as f:
	candidate_dict = CandidateDict.load_entity_from_file(f, candidate_dict)
def candidates(word):
	candidates = candidate_dict[word]
	# cand_list = {}
	# for cand_name, cand_score in sorted(candidates.items(), key=lambda x: -x[1][0]):
		# print(cand_name, cand_score)
		# cand_name = redirects[cand_name] if cand_name in redirects else cand_name
		# if (cand_name in cand_list and cand_list[cand_name] < cand_score) or cand_name not in cand_list:
		# 	cand_list[cand_name] = cand_score
	return candidates

def getETRI(text):
	host = '143.248.135.146'
	port = 33344
	
	ADDR = (host, port)
	clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	try: 
		clientSocket.connect(ADDR)
	except Exception as e:
		return None
	try:
		clientSocket.sendall(str.encode(text))
		buffer = bytearray()
		while True:
			data = clientSocket.recv(1024)
			if not data:
				break
			buffer.extend(data)
		result = json.loads(buffer.decode(encoding='utf-8'))

		return result
	except Exception as e:
		return None


def find_ne_pos(j):
	def find_in_wsd(wsd, ind):
		for item in wsd:
			if item["id"] == ind:
				return item
		print(wsd, ind)
		raise IndexError(ind)
	if j is None:
		return None
	original_text = reduce(lambda x, y: x+y, list(map(lambda z: z["text"], j["sentence"])))
	# original_text = j["sentence"]
	# print(original_text)
	# print(j)
	j["NE"] = []
	try:
		for v in j["sentence"]:
			sentence = v["text"]
			for ne in v["NE"]:
				morph_start = find_in_wsd(v["morp"],ne["begin"])
				# morph_end = find_in_wsd(v["WSD"],ne["end"])
				byte_start = morph_start["position"]
				# print(ne["text"], byte_start)
				# byte_end = morph_end["position"]+sum(list(map(lambda char: len(char.encode()), morph_end["text"])))
				byteind = 0
				charind = 0
				for char in original_text:
					if byteind == byte_start:
						ne["char_start"] = charind
						ne["char_end"] = charind + len(ne["text"])
						j["NE"].append(ne)
						break
					byteind += len(char.encode())
					charind += 1
				else:
					raise Exception("No char pos found: %s" % ne["text"])
			j["original_text"] = original_text
	except Exception as e:
		print(e)
		return None
	# print(len(j["NE"]))
	return j

def is_not_korean(char):
	return not (0xAC00 <= ord(char) <= 0xD7A3)

def change_into_crowdsourcing_form(_arg=None, text=None, file=None):
	if _arg is not None or not ((text is None) ^ (file is None)):
		raise Exception
	if text:
		if type(text) is str:
			text = [text]
	else:
		text = file.readlines()
	result = []
	c = 0
	for t in text:
		t = t.strip()
		j = find_ne_pos(getETRI(t))
		if j is None:
			print("Error: " + t)
			continue
		cs_form = {}
		cs_form["text"] = j["original_text"]
		cs_form["entities"] = []
		cs_form["fileName"] = "%d" % c
		for item in j["NE"]:
			skip_flag = False
			for prefix in ["QT", "DT"]:
				if item["type"].startswith(prefix): skip_flag = True
			if item["type"] in ["CV_RELATION", "TM_DIRECTION"] or skip_flag: continue
			
			if all(list(map(is_not_korean, item["text"]))): continue

			cs_form["entities"].append({
				"surface": item["text"],
				"candidates": candidates(item["text"]),
				"keyword": "NOT_IN_CANDIDATE",
				"start": item["char_start"],
				"end": item["char_end"],
				"ne_type": item["type"], 
				"type": "ETRI"
				})
		result.append(cs_form)
		c += 1
		if c % 1000 == 0:
			print("\r%d" % c, end = "", flush=True)
	return result

def overlap(ent1, ent2):
	s1 = ent1["start"]
	e1 = ent1["end"]
	s2 = ent2["start"]
	e2 = ent2["end"]
	return s1 <= s2 < e1 or s1 < e2 <= e1

def morph_split(morph_pos, links):
	result = []
	morph, pos = morph_pos
	for link in links:
		ne, en, sp, ep = link
		if sp <= pos < ep or pos <= sp < pos+len(morph):
			if pos < sp:
				m1 = morph[:sp-pos]
				m2 = morph[sp-pos:min(len(morph), ep-pos)] # 반[중국]적 과 같은 경우
				m3 = morph[ep-pos:] if ep < pos+len(morph) else None
				result.append([m1, None])
				result.append([m2, link])
				if m3 is not None:
					result += morph_split((m3, pos+len(m1)+len(m2)), links)
				break
			elif pos + len(morph) > ep:
				# print(morph, "/", en)
				m1 = morph[:ep-pos]
				m2 = morph[ep-pos:]
				result.append([m1, link])
				result += morph_split((m2, pos+len(m1)), links)
				break
			else:
				result.append([morph, link])
				break
	else:
		result.append([morph, None])
	return result

def change_to_conll(j, filter_emptycand=False):
	result = []
	result.append("-DOCSTART- (%s" % j["fileName"] if "fileName" in j else "TEMPVAL")
	print_flag = False
	links = []
	for entity in j["entities"]:
		redirected_entity = redirects[entity["keyword"]] if entity["keyword"] in redirects else entity["keyword"]
		if redirected_entity not in ent_form and redirected_entity != "NOT_IN_CANDIDATE":
			continue
		if filter_emptycand and redirected_entity == "NOT_IN_CANDIDATE":
			continue
		links.append((entity["surface"], entity["keyword"], entity["start"], entity["end"]))

	filter_entity = set([])
	for i1 in links:
		if i1 in filter_entity: continue
		for i2 in links:
			if i1 == i2: continue
			if i1[2] <= i2[2] < i1[3] or i1[2] < i2[3] <= i1[3]:
				# overlaps
				shorter = i1 if i1[3] - i1[2] <= i2[3] - i2[2] else i2
				filter_entity.add(shorter)
	links = list(filter(lambda x: x not in filter_entity, links))
	sentence = j["text"]
	for char in "   ":
		sentence.replace(char, " ")
	morphs = okt.morphs(sentence)
	inds = []
	last_char_ind = 0
	for item in morphs:
		ind = sentence.find(item, last_char_ind) 
		inds.append(ind)
		last_char_ind = ind+len(item)
	assert(len(morphs) == len(inds))
	last_link = None
	for morph, pos in zip(morphs, inds):
		# if "\n" in morph: continue
		# print(morph, pos)
		
		added = False
		for m, link in morph_split((morph, pos), links):
			if link is None:
				result.append(m)
				last_link = None
				continue
			last_label = result[-1][1] if len(result) > 0 and type(result[-1]) is not str else "O"
			bi = "I" if last_label != "O" and last_link is not None and link == last_link else "B"
			ne, en, sp, ep = link
			last_link = link
			assert m in ne
			result.append([m, bi, ne, en, "%s%s" % (dbpedia_prefix, en), "000", "000"])


	result = list(map(lambda x: x if type(x) is str else "\t".join(x), result))
	if result[-1] in ["", "\n"]:
		result = result[:-1]
	return result


def get_context_words(text, pos, direction, maximum_context=30):
	result = []
	ind = pos
	buf = ""
	text = text.replace("\n", " ")
	while len(result) < maximum_context and ind > 0 and ind < len(text)-1:
		ind += direction
		if text[ind] == " ":
			if len(buf) > 0:
				buf = buf[::direction]
				result.append(buf[:])
				buf = ""
			continue
		buf += text[ind]
	if len(buf) > 0:
		result.append(buf[::direction])
	if len(result) == 0:
		return "EMPTYCTXT"
	result = " ".join(result[::direction])
	# print(result)
	return result


def change_to_tsv(j, filter_emptycand=False):
	# print(fname)
	result = []
	text = j["text"]
	fname = j["fileName"]
	entity_to_text = lambda x: ",".join(["0", "0", x["entity"]])# 0을 entity id로 바꿔야 함
	entities = j["entities"]
	filter_entity = []
	for i1 in entities:
		if i1 in filter_entity: continue
		for i2 in entities:
			if i1 == i2: continue
			if overlap(i1, i2):
				# overlaps
				shorter = i1 if i1["end"] - i1["start"] <= i2["end"] - i2["start"] else i2
				filter_entity.append(shorter)
	entities = list(filter(lambda x: x not in filter_entity, entities))
	for entity in entities:
		redirected_entity = redirects[entity["keyword"]] if entity["keyword"] in redirects else entity["keyword"]
		if redirected_entity not in ent_form and redirected_entity != "NOT_IN_CANDIDATE":
			continue
		if filter_emptycand and redirected_entity == "NOT_IN_CANDIDATE":
			continue
		candidate_list = entity["candidates"]
		sp = entity["start"]
		ep = entity["end"]
		f = [fname, fname, entity["surface"], get_context_words(text, sp, -1), get_context_words(text, ep-1, 1), "CANDIDATES"]
		gold_ind = -1
		gold_sent = ""
		ind = 0
		cand_list = []
		for cand_name, cand_score in sorted(candidate_list.items(), key=lambda x: -x[1][0]):
			cand_list.append((redirects[cand_name] if cand_name in redirects else cand_name, cand_score))
		if redirected_entity == "NOT_IN_CANDIDATE": redirected_entity = "#UNK#"
		for cand_name, cand_score in cand_list:
			# print(cand_score)
			f.append(",".join([str(cand_score[1]), str(cand_score[0]), cand_name]))
			if cand_name == redirected_entity:
				gold_ind = ind
				gold_sent = f[-1]
			ind += 1
		if len(cand_list) == 0:
			f.append("EMPTYCAND")
		f.append("GE:")
		f.append("%d,%s" %(gold_ind, gold_sent) if gold_ind != -1 else "-1")
		result.append("\t".join(f))
	return result

@TimeUtil.measure_time
def prepare(*sentences, is_json=False):
	conlls = []
	tsvs = []
	cw_form = change_into_crowdsourcing_form(text=sentences) if not is_json else sentences
	for sentence in cw_form:
		try:
			conll = change_to_conll(sentence)
			tsv = change_to_tsv(sentence)
			conlls += conll
			conlls += [""]
			tsvs += tsv
		except Exception as e:
			import traceback
			traceback.print_exc()
	return cw_form, conlls, tsvs