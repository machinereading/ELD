from ...utils import jsonload
import re
def cw2conll(cw_data):

	text = cw_data["text"]
	result = []
	ent = sorted(cw_data["entities"], key=lambda x: x["start"])
	for entity in ent:
		text = text[:entity["start"]] + "<" + text[entity["start"]:entity["end"]] + ">" + text[entity["end"]:]
		for entity in ent:
			entity["start"] += 2
			entity["end"] += 2

	text = re.sub(r"[^ ㄱ-ㅎㅏ-ㅣ가-힣a-z-A-Z0-9<>]+", "", text)
	sent = []
	token_flag = False
	for token in text.split(" "):
		tokens = token.split(">")
		l = len(tokens)
		# print(tokens)
		for i, token in enumerate(tokens):
			if len(token) == 0: continue
			if i != l - 1:
				token += ">"

			# print(token)
			token_buf = []
			if token[0] == "<":
				token_flag = True
				if ">" in token:
					token_flag = False
					t0, t1 = token.split(">")
					sent.append(" ".join([str(len(sent)), t0[1:], "_", "_", "S/E"]))
					if len(t1) > 0:
						sent.append(" ".join([str(len(sent)), t1, "_", "_", "O"]))
				else:
					sent.append(" ".join([str(len(sent)), token[1:], "_", "_", "B/E"]))
				continue
			if token_flag:
				if ">" in token:
					token_flag = False
					t0, t1 = token.split(">")
					sent.append(" ".join([str(len(sent)), t0, "_", "_", "E/E"]))
					if len(t1) > 0:
						sent.append(" ".join([str(len(sent)), t1, "_", "_", "O"]))
				else:
					sent.append(" ".join([str(len(sent)), token, "_", "_", "I/E"]))
			else:
				sent.append(" ".join([str(len(sent)), token, "_", "_", "O"]))
	return sent

def sentence2conll(sentence):
	return [" ".join([i, token.text, "_", "_", "O"]) for i, token in enumerate(sentence)]

if __name__ == '__main__':
	import os
	target_dir = "corpus/crowdsourcing_formatted/"
	with open("corpus/er/crowdsourcing1.conll", "w", encoding="UTF8") as f:
		for item in os.listdir(target_dir):
			f.write("\n".join(cw2conll(jsonload(target_dir+item)))+"\n\n")
