from src.utils.datamodule.DataParser import WikiSilverCharacterParser as WikiParser
from src.ner_etri.ETRINER import getETRI
from src.utils.NERLabelUtil import decode_label
from src.utils import printfunc, progress
import itertools
import json
# if __name__ == '__main__':
# 	parser = WikiParser("corpus/wiki_silver/")
# 	result = []
# 	count = 0
# 	for _ in itertools.chain(parser.get_trainset(), parser.get_devset()):
# 		count += 1
# 	it = 0
# 	for sent, _, label in itertools.chain(parser.get_trainset(), parser.get_devset()):
# 		x = getETRI(sent)
# 		it += 1
# 		try:
# 			etri_entities = list(map(lambda x: x["NE"], x["sentence"]))
# 		except Exception:
# 			continue
# 		lbuf = ""
# 		ls = []
# 		ind = 0
# 		sin = 0
# 		for s, l in zip(sent, label):
# 			if l[0] in "BU":
# 				sin = ind
# 			if l[0] in "BILU":
# 				lbuf += s
# 			if l[0] in "LU":
# 				ls.append({"text": lbuf, "start": sin, "end": ind})
# 				lbuf = ""
# 			ind += 1
# 		result.append({"text": sent, "links": ls, "ETRI": etri_entities})
# 		printfunc(progress(it, count, 50)+", %d" % it)
# 	with open("corpus/wiki_etri_parsed.txt", "w", encoding="UTF8") as f:
# 		json.dump(result, f, ensure_ascii=False, indent="\t")

sent = "토비 레너드 무어(, 1981년 4월 21일 ~ )는 오스트레일리아 출신의 배우로, 넷플릭스의 드라마 《데어데블》에서 악당 킹핀의 비서 제임스 웨슬리 역으로 출연하였다."
x = getETRI(sent)
print(list(map(lambda x: x["NE"], x["sentence"])))