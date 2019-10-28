from ...utils import getETRI
from ... import GlobalValues as gl
from functools import reduce
import random
def find_ne_pos(j):
	def find_in_wsd(wsd, ind):
		for item in wsd:
			if item["id"] == ind:
				return item
		print(wsd, ind)
		raise IndexError(ind)

	if j is None:
		gl.logger.debug("ETRI result is None")
		return None
	original_text = reduce(lambda x, y: x + y, list(map(lambda z: z["text"], j["sentence"])))
	# original_text = j["sentence"]
	# print(original_text)
	# print(j)
	j["NE"] = []
	try:
		for v in j["sentence"]:
			for ne in v["NE"]:
				morph_start = find_in_wsd(v["morp"], ne["begin"])
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

def text_to_etri(text):
	return find_ne_pos(getETRI(text))

def etri_to_ne_dict(etri):
	if etri is None:
		return None
	cs_form = {}
	cs_form["text"] = etri["original_text"] if "original_text" in etri else etri[
		"text"]
	cs_form["entities"] = []
	# cs_form["fileName"] = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in
	#                               range(7)) if "fileName" not in ne_marked_dict or ne_marked_dict[
	# 	"fileName"] == "" else etri["fileName"]
	entities = etri["NE"] if "NE" in etri else etri["entities"]
	for item in entities:
		skip_flag = False
		if "type" in item:
			for prefix in ["QT", "DT"]:  # HARD-CODED: ONLY WORKS FOR ETRI TYPES
				if item["type"].startswith(prefix): skip_flag = True
			if item["type"] in ["CV_RELATION", "TM_DIRECTION"] or skip_flag: continue
		surface = item["text"] if "text" in item else item["surface"]
		# if "keyword" in item or "entity" in item:
		# 	keyword = "NOT_IN_CANDIDATE" if predict else (item["keyword"] if "keyword" in item else item["entity"])
		# else:
		# 	keyword = "NOT_IN_CANDIDATE"
		# if keyword == "NOT_AN_ENTITY":
		# 	keyword = "NOT_IN_CANDIDATE"
		# if "dark_entity" in item:
		# 	keyword = "DARK_ENTITY"
		start = item["char_start"] if "char_start" in item else item["start"]
		end = item["char_end"] if "char_end" in item else item["end"]
		# if not any(list(map(KoreanUtil.is_korean_character, surface))): continue  # non-korean filtering

		cs_form["entities"].append({
			"surface"   : surface,
			# "candidates": self.surface_ent_dict[surface],
			# "answer"    : keyword,
			"start"     : start,
			"end"       : end,
			"ne_type"   : item["type"] if "type" in item else ""
		})
	return cs_form
