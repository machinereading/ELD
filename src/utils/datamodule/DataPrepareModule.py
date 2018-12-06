# Corpus file을 sentence, label로 분리해 주는 generator function의 집합
# 반드시 sentence, label을 yield하는 함수로 만들 것.
import re
import traceback
import KoreanUtil
import json
import configparser
from BracketSeparatedSentence import SeparatedSentence
import random
parser = KoreanUtil.t
config = configparser.ConfigParser()
config.read("ner.ini")
tag_mode = config["tagging"]["tag_type"]

def sentence_morph_wrap(sent, labels):
	pos_tags = parser.pos(sent)
	char_index = 0
	morph_labels = []
	inside_label = False
	for morph, pos in pos_tags:
		begin_flag = False
		end_flag = False
		label_type = ""
		for char in morph:
			# print(char, sent[char_index], len(sent), char_index)
			label = labels[char_index]
			if label[0] != "O":
				label_type = label.split("/")[-1]
			if label[0] == "B":
				begin_flag = True
				inside_label = True
			if label[0] == "L":
				end_flag = True
				inside_label = False
			char_index += 1
			if char_index < len(sent) and sent[char_index] == " ":
				char_index += 1

		if begin_flag and end_flag:
			morph_labels.append("U/%s" % label_type)
		elif begin_flag:
			morph_labels.append("B/%s" % label_type)
		elif end_flag:
			morph_labels.append("L/%s" % label_type)
		elif inside_label:
			morph_labels.append("I/%s" % label_type)
		else:
			morph_labels.append("O")
	return pos_tags, morph_labels

def morph_wrapper(generator, args=None):
	# if args is None or not hasattr(args, "morph_type"):
	# 	parser = cls.t
	# else:
	# 	parser = cls.p_dict[args.morph_type]
	
	for sent, labels in generator:
		yield sentence_morph_wrap(sent, labels)

def morph_wrapper_with_original_sentence(generator):
	for sent, labels in generator:
		yield (sent, *sentence_morph_wrap(sent, labels))


def separate_sentence(sentence):
	result = []
	buf = ""
	pm = 0
	for char in sentence:
		if buf == "" and char == " ": continue
		buf += char
		if char == "(": pm = -1
		if pm == -1 and char == ")": pm = 0
		if pm == 0 and char == "다": pm = 1
		if pm == 1 and char == ".":
			pm = 0
			result.append(buf)
			buf = ""
	return result





def separate(corpus_fn, corpus_file):
	for sent, labels in corpus_fn(corpus_file):
		yield SeparatedSentence(sent, labels)

def korean_ner(hclt_format_file, include_tag=True):
	sentences = []
	tagdict = {"LC": "L", "PS": "P", "OG": "O", "DT": "T", "TI": "T"}
	for line in hclt_format_file.readlines():
		if line.startswith(";"):
			ap = line.strip()[2:]
			sentences.append(ap)
		if line.startswith("$"):
			temp = []
			labeltemp = ""
			pm = 0
			s = line.strip()[1:]
			for char in s:
				# print(char, end="")
				if char == '<' and pm == 0:
					pm = 1
					continue
				if char == ">" and pm == 1:
					pm = 0
					x = labeltemp.split(":")
					labeltemp = ":".join(x[:-1])
					ty = tagdict[x[-1]]
					if len(labeltemp) == 1:
						temp.append("U/%s" % ty)
						labeltemp = ""
						continue
					tags = (("B/%s " % ty) +(("I/%s " % ty)*(len(labeltemp)-2))+("L/%s" % ty)).split(" ") if tag_mode.lower() == "bilou" else (("B/%s " % ty) +(("I/%s " % ty)*(len(labeltemp)-1))).split(" ")
					for item in tags:
						temp.append(item)
					labeltemp = ""
					continue
				
				if pm == 0:
					temp.append("O")
				elif pm == 1:
					labeltemp += char
			yield sentences[-1], temp
			sentences = []

def hclt_jamo(corpus_file, include_tag=True):
	sentences = []
	tagdict = {"LC": "L", "PS": "P", "OG": "O", "DT": "T", "TI": "T"}
	for line in corpus_file.readlines():
		if line.startswith(";"):
			ap = line.strip()[2:]
			ap = KoreanUtil.decompose_sent(ap)
			sentences.append(ap)
		if line.startswith("$"):
			temp = []
			labeltemp = ""
			pm = 0
			s = line.strip()[1:]
			s = KoreanUtil.decompose_sent(s)
			for char in s:
				# print(char, end="")
				if char == '<' and pm == 0:
					pm = 1
					continue
				if char == ">" and pm == 1:
					pm = 0
					x = labeltemp.split(":")
					labeltemp = ":".join(x[:-1])
					ty = tagdict[x[-1]]
					if len(labeltemp) == 1:
						tag = "U" if tag_mode.lower() == "bilou" else "B"
						temp.append("%s/%s" % (tag, ty))
						labeltemp = ""
						continue
					tags = (("B/%s " % ty) +(("I/%s " % ty)*(len(labeltemp)-2))+("L/%s" % ty)).split(" ") if tag_mode.lower() == "bilou" else (("B/%s " % ty) +(("I/%s " % ty)*(len(labeltemp)-1))).split(" ")
					for item in tags:
						temp.append(item)
					labeltemp = ""
					continue
				
				if pm == 0:
					temp.append("O")
				elif pm == 1:
					labeltemp += char
			yield sentences[-1], temp
			sentences = []

def hclt_morph(corpus_file):
	mp = []
	labels = []
	lt = "O"
	tagdict = {"LC": "L", "PS": "P", "OG": "O", "DT": "T", "TI": "T"}
	for line in corpus_file.readlines():
		if line[0] in ";$": continue
		if len(line.strip()) == 0:
			yield mp, labels
			mp = []
			labels = []
			lt = "O"
			continue
		_, morph, pos, tag = line.strip().split("\t")
		ltype = lt.split("/")[-1]
		if tag[0] in "BO":
			if len(labels) > 0 and labels[-1][0] == "I":
				labels[-1] = "L/"+ltype
			if len(labels) > 0 and labels[-1][0] == "B":
				labels[-1] = "U/"+ltype
		if tag[0] == "B":
			tag = "B/"+tagdict[tag.split("_")[-1]]
		if tag[0] == "I":
			tag = "I/"+lt[-1]
		labels.append(tag)
		mp.append((morph, pos))

def corpus(corpus_file, separate=False):
	for line in corpus_file.readlines():
		# while True:
		# 	try:
		# 		si = line.index("<<")
		# 		ei = line.index(">>", si)
		# 	except Exception:
		# 		break
		# 	line = line[:si]+line[ei+3:] # >> 뒤에는 항상 공백이 하나 있더라?
		if len(line.strip()) == 0 or line.strip() =="<&doc&>": continue
		skip = False
		sentences = separate_sentence(line)
		if separate:
			sep = SeparatedSentence(line)
			sentences = [sep.removed_sentence] + sep.brackets
		for line in sentences:
			ind = []
			while True:
				try:
					si = line.index("[[[")
					ei = line.index(">>", si)
				except Exception:
					break
				
				txt = line[si+3:ei]
				try:
					en, ty = txt.split("]]]<<")
					en = en.split("|")[0]
					ty = ty.split("|")[-1]
					if "[" in en:
						skip = True
				except Exception:
					skip = True
					break
				if en.endswith(" "): en = en[:-1]
				ind.append((si, len(en), ty))
				line = line[:si]+en+line[ei+3:]
			label = []
			if skip: continue
			lastentity = None
			for s, l, t in ind:
				if lastentity is not None and \
				   lastentity[-1][0] == "T" and \
				   t[0] == "T" and \
				   s - (lastentity[0] + lastentity[1]) == 1 and \
				   lastentity[0] + lastentity[1] < len(line) and \
				   line[(lastentity[0] + lastentity[1])] in " ":
					label[-1] = "I/T"
					label.append("I/T")
					tags = (("I/%s " % t[0])+(("I/%s " % t[0]) * (l-2))+("L/%s" % t[0])).split(" ") if tag_mode.lower() == "bilou" else (("B/%s " % t[0])+(("I/%s " % t[0]) * (l-1))).split(" ")
					for c in tags:
						label.append(c)
				else:
					for _ in range(s-len(label)):
						label.append("O")
					if l == 1:
						label.append("U/%s" % t[0])
					else:
						tags = (("B/%s " % t[0])+(("I/%s " % t[0]) * (l-2))+("L/%s" % t[0])).split(" ") if tag_mode.lower() == "bilou" else (("B/%s " % t[0])+(("I/%s " % t[0]) * (l-1))).split(" ")
						for c in tags:
							label.append(c)
				lastentity = (s,l,t)
			l = line.strip()
			for _ in range(len(l) - len(label)):
				label.append("O")
			yield l, label



	

def premade(corpus_file):
	i = 0
	sent = []
	label = []
	for line in corpus_file.readlines():
		if len(line.strip()) == 0: continue
		if i % 2 == 0:
			sent = line.strip().split("/sep/")
		else:
			label = line.strip().split("/sep/")
			if len(sent) == 0 or len(label) == 0: continue
			yield sent, label
		i += 1
		

def etri_golden(corpus_file):
	label_func = lambda x: x[0] if x[0] not in ["T", "D"] else "M"
	for sentence in corpus_file.readlines():
		s = sentence.strip()
		pm = 0
		senttemp = ""
		labeltemp = ""
		labels = []
		for c in s:
			if c == "<":
				pm = 1
				continue
			if pm == 1 and c == ">":
				pm = 0
				ll = labeltemp.split(":")
				surface = ":".join(ll[:-1])
				ty = ll[-1]
				senttemp += surface
				labeltype = label_func(ty)
				if len(surface) == 1:
					labels.append("U/%s" % labeltype)
				else:
					for item in (("B/%s " % labeltype)+(("I/%s " % labeltype) * (len(surface)-2)) + ("L/%s" % labeltype)).split(" "):
						labels.append(item)
				labeltemp = ""
				continue
			if pm == 1:
				labeltemp += c
			else:
				senttemp += c
				labels.append("O")
		yield senttemp, labels

def wikipedia_golden(corpus_file):
	for sentence in corpus_file.readlines():
		if len(sentence.strip()) == 0: continue
		pm = 0
		sent = ""
		lt = ""
		labels = []
		for c in sentence.strip():
			if c == "[":
				pm += 1
				continue
			if c == "]":
				pm = 0
				try:
					surface, ty = lt.split(";")
				except Exception:
					print(sentence)
					import sys
					sys.exit(1)
				if len(surface) == 1:
					labels.append("U/%s" % ty)
				else:
					for item in (("B/%s " % ty)+(("I/%s " % ty) * (len(surface)-2)) + ("L/%s" % ty)).split(" "):
						labels.append(item)
				lt = ""
				sent += surface
				continue
			if pm == 0:
				sent += c
				labels.append("O")
			else:
				lt += c
		yield sent, labels

def crowdsourcing_golden(corpus_file):
	js = json.load(corpus_file)
	for j in js:
		sent = j["plainText"]
		labels = []
		lastentity = None
		for item in j["entities"]:
			s = item["st_mention"]
			e = item["en_mention"]
			if len(item["keyword"]) == 2:
				continue
			if lastentity is not None and \
			   lastentity["ne_type"][0] == "T" and \
			   item["ne_type"][0] == "T" and \
			   item["st_mention"] - lastentity["en_mention"] == 1 and \
			   sent[lastentity["en_mention"]] == " ":
				labels[-1] = "I/T"
				labels.append("I/T")
				for c in (("I/%s " % ne_type)+(("I/%s " % ne_type)*(e-s-2)+("L/%s" % ne_type))).split(" "):
					labels.append(c)
			else:
				for _ in range(s - len(labels)):
					labels.append("O")
				ne_type = item["ne_type"][0]
				if ne_type not in "PLOT": ne_type = "M"
				if e - s == 1:
					labels.append("U/%s" % ne_type)
				else:
					for c in (("B/%s " % ne_type)+(("I/%s " % ne_type)*(e-s-2)+("L/%s" % ne_type))).split(" "):
						labels.append(c)
			lastentity = item
		for _ in range(len(sent) - len(labels)):
			labels.append("O")
		yield sent, labels

def crowdsourcing_fixed(corpus_file):
	j = json.load(corpus_file)
	for s in j:
		sentence = s["entity_tagged_text"]
		plain_text = s["plain_text"]
		labelbuf = []
		entitybuf = ""
		pm = 0
		for char in sentence:
			if char == "[":
				pm = 1
				continue
			if char == "]":
				pm = 0
				x = entitybuf.split(":")
				entity_text = ":".join(x[:-1])
				entity_type = x[-1]
				if len(entity_text) == 1:
					labelbuf.append("U/%s" % entity_type)
				else:
					for c in (("B/%s " % entity_type)+(("I/%s " % entity_type)*(len(entity_text)-2)+("L/%s" % entity_type))).split(" "):
						labelbuf.append(c)
				entitybuf = ""
				continue


			if pm == 0:
				labelbuf.append("O")
			else:
				entitybuf += char
		yield plain_text, labelbuf

def raw(f, separate=False):
	for sentence in f.readlines():
		# print(sentence)
		pm = 0
		plain_text = ""
		entitybuf = ""
		labelbuf = []
		for char in sentence.strip():
			if char == "[":
				pm = 1
				continue
			if char == "]":
				pm = 0
				x = entitybuf.split(":")
				entity_text = ":".join(x[:-1])
				plain_text += entity_text
				entity_type = x[-1]
				if len(entity_text) == 1:
					labelbuf.append("U/%s" % entity_type)
				else:
					for c in (("B/%s " % entity_type)+(("I/%s " % entity_type)*(len(entity_text)-2)+("L/%s" % entity_type))).split(" "):
						labelbuf.append(c)
				entitybuf = ""
				continue


			if pm == 0:
				labelbuf.append("O")
				plain_text += char
			else:
				entitybuf += char
		yield plain_text, labelbuf


def convert_to_conll(generator, wf, max=None, dropout=0):
	tagdict = {"L": "E", "U": "S"}
	typedict = {"P": "PER", "L": "LOC", "O": "ORG", "M": "MISC", "T": "TIME"}
	write = 0

	for sentence, label in generator:
		i = 0
		buf = []
		if all([x == "O" for x in label]): continue
		if random.random() < dropout: continue

		try:
			for c, l in zip(sentence, label):
				morph, pos = c
				if l == "O":
					buf.append("{} {} {} _ {}\n".format(i, morph, pos, l))
				else:
					tag, type = l.split("/")
					tag = tagdict[tag] if tag in tagdict else tag
					type = typedict[type]
					buf.append("{} {} {} _ {}-{}\n".format(i, morph, pos, tag, type))
				i += 1
		except Exception:
			continue
		for item in buf:
			wf.write(item)
		# map(lambda x: wf.write(x), buf)
		wf.write("\n")
		write += 1
		if max and max < write:
			return

def convert_hclt_to_conll(f, wf):
	count = 0
	tagdict = {"LC": "LOC", "PS": "PER", "OG": "ORG", "DT": "TIME", "TI": "TIME"}
	tagtype = None
	for line in f.readlines():
		if len(line.strip()) == 0:
			count = 0
			wf.write("\n")
			continue
		if line[0] in "$;":
			continue
		n, c, p, t = line.strip().split("\t")
		if t[0] == "B":
			tagtype = tagdict[t.split("_")[1]]
			t = "B-%s" % tagtype
		if t[0] == "I":
			t = "I-%s" % tagtype
		wf.write("{} {} {} _ {}\n".format(count, c, p, t))
		count += 1




if __name__ == '__main__':
	import os
	from NERUtil import wrap_entities_in_text, decode_label, decode_morph_labels
	import sys
	result = []
	i = 0
	wiki = "corpus/entityTypeTaggedText_fixed.txt"
	hclt_train = "corpus/koreanner/original/train.txt"
	hclt_dev = "corpus/koreanner/original/dev.txt"

	with open("corpus/crowdsourcing_fix.json", encoding="UTF8") as f, open("crowdsourcing_morph.conll", "w", encoding="UTF8") as wf:
		convert_to_conll(morph_wrapper(crowdsourcing_fixed(f)), wf)