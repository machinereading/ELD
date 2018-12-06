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

def change_to_conll(js):
	result = []
	# result.append("-DOCSTART- (%s" % js["fileName"] if "fileName" in js else "TEMPVAL")
	# print_flag = js["fileName"] in ["샤오미"]
	print_flag = False
	links = []
	for entity in js["entities"]:
		links.append((entity["surface"], entity["start"], entity["end"]))

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
	# for ne in j["addLabel"]:
	# 	links.append((ne["keyword"], ne["candidates"][ne["answer"]]["entity"] if "candidates" in ne and ne["answer"] >= 0 else ne["keyword"].replace(" ", "_"), ne["startPosition"], ne["endPosition"]))
	# for wikilink in j["entities"]:
	# 	links.append((wikilink["surface"], wikilink["keyword"], wikilink["st"], wikilink["en"]))
	sentence = js["text"]
	for char in "  ":
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
	last_label = "O"
	for morph, pos in zip(morphs, inds):
		# if "\n" in morph: continue
		# print(morph, pos)
		
		added = False
		for m, link in morph_split((morph, pos), links):
			if link is None:
				result.append([m, "O"])
				last_link = None
				continue
			last_label = result[-1][1] if len(result) > 0 else "O"

			# last_en = result[-1][3] if len(result) > 0 and type(result[-1]) is not str else ""
			# last_sf = result[-1][2] if len(result) > 0 and type(result[-1]) is not str else ""
			bi = "I" if last_label != "O" and last_link is not None and link == last_link else "B"
			ne, sp, ep = link
			last_link = link
			result.append([m, bi])


	result = list(map("\t".join(x), result))
	if result[-1] in ["", "\n"]:
		result = result[:-1]
	return "\n".join(result)