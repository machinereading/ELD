from abc import ABC, abstractmethod, abstractproperty
from ... import GlobalValues as gl
from konlpy.tag import Okt
import os
import json
import numpy as np
import traceback
def _separate_sentence(sentence):
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

def _decode_label_default(sentence, labels, type_marker="-"):
	result = []
	buf = []
	parse_mode = 0
	type_buf = ""
	last = "O"
	sin = 0
	ind = 0
	for s, l in zip(sentence, labels):

		if type(l) in (np.int32, np.int64, int):
			l = gl.ner.label_dict_inv[l] if l in gl.ner.label_dict_inv else "O"
		if l[0] == "B":
			if last[0] != "O":
				if type_buf != "":
					result.append({
						"name": " ".join(buf),
						"type": type_buf,
						"start_index": sin,
						"end_index": sin + len(buf),
						"score": 0,
						"index": len(result)+1
						})
				buf = []
			type_buf = l.split(type_marker)[-1]
			parse_mode = 1
			sin = ind
		if parse_mode == 1 and l[0] == "O":
			result.append({
				"name": " ".join(buf),
				"type": type_buf,
				"start_index": sin,
				"end_index": sin + len(buf),
				"score": 0,
				"index": len(result)+1
				})
			buf = []
			parse_mode = 0

		if parse_mode == 1:
			buf.append(s)

		last = l
		ind += 1
	# print(sentence, result)
	return result



def change_into_conll(sentence):
	pass

class AbstractDataParser(ABC):
	def __init__(self, data_dir):
		self.data_dir = data_dir
		self.pos = Okt()
		if not self.data_dir.endswith("/"): self.data_dir += "/"
		
		if not os.path.isfile(self.data_dir+"__loader.json"):
			self.file_dict = {"train": [], "dev": []}
			iter_count = 0
			if len(os.listdir(self.data_dir)) == 1:
				# separate train and dev set in 9:1 ratio
				print("Writing train and dev set...")
				with open(self.data_dir+os.listdir(self.data_dir)[0], encoding="UTF8") as f, \
					 open(self.data_dir+"train.txt", "w", encoding="UTF8") as tf, \
					 open(self.data_dir+"dev.txt", "w", encoding="UTF8") as df:
					sent_iter_count = 0
					for unit in self.unit_generator(f):
						w = unit if type(unit) is str else "\n".join(unit)
						sent_iter_count += 1
						if sent_iter_count % 10 == 0:
							df.write(w+"\n\n")
						else:
							tf.write(w+"\n\n")


				print("Done!")
				self.file_dict["train"].append("train.txt")
				self.file_dict["dev"].append("dev.txt")
				with open(data_dir+"__loader.json", "w", encoding="UTF8") as f:
					json.dump(self.file_dict, f, ensure_ascii=False, indent="\t")

			else:
				print("Dividing train and dev set...")
				for fname in os.listdir(self.data_dir):
					if "train" in fname:
						self.file_dict["train"].append(fname)
						continue
					if "dev" in fname or "eval" in fname:
						self.file_dict["dev"].append(fname)
						continue
					if iter_count % 10 == 0:
						self.file_dict["dev"].append(fname)
					else:
						self.file_dict["train"].append(fname)
					iter_count += 1
			with open(data_dir+"__loader.json", "w", encoding="UTF8") as f:
				json.dump(self.file_dict, f, ensure_ascii=False, indent="\t")
		else:
			with open(data_dir+"__loader.json", encoding="UTF8") as f:
				self.file_dict = json.load(f)
	
	def get_trainset(self):
		for input_file in self.file_dict["train"]:
			try:
				with open(self.data_dir+input_file, encoding="UTF8") as f:
					for unit in self.unit_generator(f):
						data = self.unit_to_data(unit)
						if data is not None: yield data
			except Exception as e: print(e)  

	def get_devset(self):
		for input_file in self.file_dict["dev"]:
			try:
				with open(self.data_dir+input_file, encoding="UTF8") as f:
					for unit in self.unit_generator(f):
						data = self.unit_to_data(unit)
						if data is not None: yield data
			except Exception as e:
				traceback.print_exc()

	

	@abstractmethod
	def unit_generator(self, input_file):
		return None

	@abstractmethod
	def unit_to_data(self, unit_data):
		return None

	@abstractmethod
	def decode_label(self, sentence, labels):
		return None


class WikiSilverCharacterParser(AbstractDataParser):
	def unit_generator(self, input_file):
		for line in input_file.readlines():
			line = line.strip()
			if len(line) == 0 or line =="<&doc&>": continue
			sentences = _separate_sentence(line)
			for sentence in sentences:
				yield sentence

	def unit_to_data(self, unit):
		ind = []
		line = unit
		skip = False
		tag_mode = "BILOU"
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
		if skip: return None
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
		return l, None, label # _ for pos tag

	def decode_label(self, sentence, labels):
		return _decode_label_default(sentence, labels)

class HCLTMorphParser(AbstractDataParser):
	def unit_generator(self, input_file):
		buf = []
		for line in input_file.readlines():
			line = line.strip()
			if len(line) == 0: 
				yield buf
				buf = []
				continue
			buf.append(line)

	def unit_to_data(self, unit):
		unit = unit[2:]
		sbuf = []
		pbuf = []
		lbuf = []
		for _, s, p, l in map(lambda x: x.split("\t"), unit):
			sbuf.append(s)
			pbuf.append(p)
			lbuf.append(l)
		return sbuf, pbuf, lbuf
	
	def decode_label(self, sentence, labels):
		return _decode_label_default(sentence, labels, type_marker="_")


class HCLTCharacterParser(AbstractDataParser):
	pass

class CoNLLParser(AbstractDataParser):
	def unit_generator(self, input_file):
		buf = []
		for line in input_file.readlines():
			line = line.strip()
			if len(line) == 0: 
				yield buf
				buf = []
				continue
			buf.append(line)

	def unit_to_data(self, unit):
		sbuf = []
		pbuf = []
		lbuf = []
		for line in unit:
			_, morph, pos, _, label = line.split(" ")
			if len(morph) > 50: return None
			sbuf.append(morph)
			pbuf.append(pos)
			lbuf.append(label)

		return sbuf, pbuf, lbuf

	def decode_label(self, sentence, labels):
		return _decode_label_default(sentence, labels)

class NaverContestParser(AbstractDataParser):
	def unit_generator(self, input_file):
		buf = []
		for line in input_file.readlines():
			line = line.strip()
			if len(line) == 0: 
				yield buf
				buf = []
				continue
			buf.append(line)

	def unit_to_data(self, unit):
		sbuf = []
		pbuf = []
		lbuf = []
		for line in unit:
			_, morph, label = line.split("\t")
			sbuf.append(morph)
			pbuf.append(None)
			label = "_".join(label.split("_")[::-1])
			lbuf.append("O" if label == "-" else label)
		return sbuf, pbuf, lbuf

	def decode_label(self, sentence, labels):
		return _decode_label_default(sentence, labels, type_marker="_")

class CrowdsourcingParser(AbstractDataParser):
	def unit_generator(self, input_file):
		yield json.load(input_file)

	def morph_split(self, morph_pos, links):
		result = []
		morph, pos = morph_pos
		for link in links:
			sp = link["start"]
			ep = link["end"]
			if sp <= pos < ep or pos <= sp < pos+len(morph):
				if pos < sp:
					m1 = morph[:sp-pos]
					m2 = morph[sp-pos:min(len(morph), ep-pos)] # 반[중국]적 과 같은 경우
					m3 = morph[ep-pos:] if ep < pos+len(morph) else None
					result.append([m1, None])
					result.append([m2, link])
					if m3 is not None:
						result += self.morph_split((m3, pos+len(m1)+len(m2)), links)
					break
				elif pos + len(morph) > ep:
					# print(morph, "/", en)
					m1 = morph[:ep-pos]
					m2 = morph[ep-pos:]
					result.append([m1, link])
					result += self.morph_split((m2, pos+len(m1)), links)
					break
				else:
					result.append([morph, link])
					break
		else:
			result.append([morph, None])
		return result


	def unit_to_data(self, unit):
		text = unit["text"]
		pos_analyzed = self.pos.pos(text.replace("\n", " "))
		entities = unit["entities"]
		lastind = 0
		inds = []
		for morph, pos in pos_analyzed:
			ind = text.index(morph, lastind)
			inds.append(ind)
			lastind = ind+len(morph)
		assert(len(inds) == len(pos_analyzed))
		m_batch = []
		p_batch = []
		l_batch = []
		for mp, ind in zip(pos_analyzed, inds):
			added = False
			morph, pos = mp
			for m, link in self.morph_split((morph, ind), entities):
				
				if link is None:
					m_batch.append(m)
					p_batch.append(pos)
					
					if len(l_batch) > 0: 
						if l_batch[-1][0] == "O": pass
						else: 
							l = l_batch[-1].split("-")[0]
							ty = l_batch[-1].split("-")[1]
							l_batch[-1] = ("E-%s" if l == "I" else "S-%s") % ty

					last_link = None
					l_batch.append("O")
					continue
				last_label = l_batch[-1] if len(l_batch) > 0 else "O"
				bi = "I-%s" % link["dataType"] if last_label != "O" and last_link is not None and link == last_link else "B-%s" % link["dataType"]
				last_link = link
				m_batch.append(m)
				p_batch.append(pos)
				l_batch.append(bi)
		return m_batch, p_batch, l_batch

	def decode_label(self, sentence, labels):
		return _decode_label_default(sentence, labels)



if __name__ == '__main__':
	x = NaverContestParser("corpus/naver_contest/")
	for s, p, l in x.get_devset():
		for ss, pp, ll in zip(s,p,l):
			print(ss, pp, ll)
		print(x.decode_label(s, l))