


def convert_tag_mode(label_seq, mode):
	if mode.lower() not in ["bilou", "bio", "bioes"]: raise Exception("Invalid mode")
	
def decode_label(sentence, label_seq, score=None):
	"""
	decode label sequence
	from given sentence and labels, extract entity mentions
	"""
	entities = []
	temp = ""
	pm = 0
	if score is None:
		score = [None] * len(label_seq)
	label = list(map(label_int_to_str, label_seq))
	ind = 0
	sin = 0
	sc = 0
	for c, l, s in zip(sentence, label, score):
		if l[0] == "U":
			entities.append({"name": sentence[ind], "type": l[-1], "start_index": ind, "end_index": ind+1, "score": float(sc / (ind-sin+1)) if s is not None else 0})
			ind += 1
			continue
		if l[0] == "B":
			pm += 1
			sin = ind
			temp += c
			if s is not None:
				sc += s[label_no(l)]
		if pm == 1 and l[0] == "I":
			temp += c
			if s is not None:
				sc += s[label_no(l)]
		if pm == 1 and l[0] == "L":
			temp += c
			if s is not None:
				sc += s[label_no(l)]
			entities.append({"name": temp, "type": l[-1], "start_index": sin, "end_index": ind+1, "score": float(sc / (ind-sin+1))}) # surface, type, start index, end index, score
			temp = ""
			pm = 0
		ind += 1
	entities.sort(key=lambda x: x["start_index"])
	i = 1
	for item in entities:
		item["index"] = i
		i += 1
	return entities