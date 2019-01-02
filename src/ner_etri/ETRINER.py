import urllib.request
from urllib.parse import urlencode
from urllib.parse import quote
import json
import pprint
import socket
import struct
import itertools
from ..utils.datamodule.DataParser import WikiSilverCharacterParser as WikiParser
def getETRI(text):
	host = '143.248.135.146'
	port = 33344
	
	ADDR = (host, port)
	client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	try:
		client_socket.connect(ADDR)
	except Exception as e:
		return None
	try:
		client_socket.sendall(str.encode(text))
		#clientSocket.sendall(text.encode('unicode-escape'))
		#clientSocket.sendall(text.encode('utf-8'))
		buffer = bytearray()
		while True:
			data = client_socket.recv(1024)
			if not data:
				break
			buffer.extend(data)
		result = json.loads(buffer.decode(encoding='utf-8'))
		return result
	except Exception as e:
		return None

if __name__ == '__main__':
	parser = WikiParser("corpus/wiki_silver/")
	with open("corpus/wiki_etri_parsed.txt", "w", encoding="UTF8") as f:
		for sent, _, _ in itertools.chain(parser.get_trainset(), parser.get_devset()):
			print(getETRI(sent))