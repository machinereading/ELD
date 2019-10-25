import sys
from src.utils import jsonload

result_file = sys.argv[1]
j = jsonload(result_file)
while True:
	target = input("Input: ")
	tp = 0
	fp = 0
	fn = 0
	for item in j["result"]:
		ent = item["Entity"]
		pred = item["EntPred"]
		if ent == target and pred == target:
			tp += 1
		elif ent == target:
			fn += 1
		elif pred == target:
			fp += 1
	print(tp, fp, fn)
