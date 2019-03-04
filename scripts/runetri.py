from src.utils import readfile, jsondump
from tqdm import tqdm
import socket
from functools import reduce
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
	j["entities"] = []
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
						ne["start"] = charind
						ne["end"] = charind + len(ne["text"])
						j["entities"].append(ne)
						break
					byteind += len(char.encode())
					charind += 1
				else:
					raise Exception("No char pos found: %s" % ne["text"])
			j["text"] = original_text
	except Exception as e:
		print(e)
		return None
	# print(len(j["NE"]))
	return j

buf = []
i = 0

def job(sentences, ind):
	buf = []
	for sent in sentences:
		try:
			buf.append(find_ne_pos(getETRI(sent)))
		except:
			pass
	jsondump(buf, "news_etri_parsed%d.json" % ind)

import threading
with open("corpus/news_parsed.txt", encoding="UTF8") as f:
	l = len(f.readlines())
for sentence in tqdm(readfile("/home/sangha/Corpus/brochette/news_parsed.txt"), total=l):
	try:
		_, _, _, sent = sentence.split("\t")
		buf.append(sent)
	except:
		pass
	if len(buf) == 1000:
		job(buf, i)
		i += 1
		buf = []

