import json
import os
import pickle
import socket
from multiprocessing.pool import Pool
import multiprocessing

from .AbstractArgument import AbstractArgument
from .Embedding import Embedding
# not used anymore - use tqdm instead
def progress(curprog, total, size=10):
	if curprog < 0 or total < 0 or size < 0:
		return ""
	a = int((curprog / total) * size)
	return "[" + ("*" * a) + (" " * (size - a)) + "]"

def printfunc(s):
	print("\r" + s, end="", flush=True)

# useful macros
def jsonload(fname):
	with open(fname, encoding="UTF8") as f:
		return json.load(f)

def jsondump(obj, fname):
	with open(fname, "w", encoding="UTF8") as f:
		json.dump(obj, f, ensure_ascii=False, indent="\t")

def readfile(fname):
	with open(fname, encoding="UTF8") as f:
		for line in f.readlines():
			yield line.strip()

def writefile(iterable, fname, processor=lambda x: x):
	with open(fname, "w", encoding="UTF8") as f:
		for item in iterable:
			f.write(processor(item) + "\n")

def pickleload(fname):
	with open(fname, "rb") as f:
		result = pickle.load(f)
	return result

def pickledump(obj, fname):
	with open(fname, "wb") as f:
		pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def split_to_batch(l, batch_size=100):
	return list(filter(lambda x: len(x) > 0, [l[x * batch_size:x * batch_size + batch_size] for x in range(len(l) // batch_size + 1)]))

# result = []
# temp = []
# for item in l:
# 	temp.append(item)
# 	if len(temp) == batch_size:
# 		result.append(temp[:])
# 		temp = []
# result.append(temp)
# return result

def split_to_equal_size(l, num):
	k = len(l) // num
	return [l[x * k:(x + 1) * k] for x in range(num + 1)]

def one_hot(i, total):
	i = int(i)
	result = [0 for _ in range(total)]
	result[i] = 1
	return result

inv_dict = lambda x: {v: k for k, v in x.items()}

def getETRI(text):
	if text == "": return None
	from .. import GlobalValues as gl
	host = '143.248.135.146'
	port = 44444
	ADDR = (host, port)
	clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	try:
		clientSocket.connect(ADDR)
	except Exception:
		gl.logger.warning("ETRI connection failed")
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
	except Exception:
		gl.logger.warning("ETRI connection lost")
		return None
	finally:
		clientSocket.close()

def dictload(k, d, default=0):
	return d[k] if k in d else default

def work_in_thread(fn, iterable, workers=multiprocessing.cpu_count() - 1):
	with Pool(workers) as p:
		result = p.map(fn, iterable)
	return result

def diriter(path):
	for p, d, f in os.walk(path):
		for ff in f:
			yield "/".join([p, ff])
