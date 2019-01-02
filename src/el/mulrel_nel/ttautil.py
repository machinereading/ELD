import re
def preprocess(line):
	return re.sub("\[.*\]", "", line)
def change_line(line):
	line = ",".join(line.strip().split(",")[:-1]).strip('"').strip()
	# line = line.strip().strip('"').strip()
	line = preprocess(line)
	remove_text = ["<e1>", "</e1>", "<e2>", "</e2>"]
	for rt in remove_text:
		line = line.replace(rt, "")
	return line

def change_file(f):
	result = set([])
	for line in f.readlines():
		result.add(change_line(line))
	return result

def overlaps(e1, e2):
	return e1[0] <= e2[0] < e1[1] or e1[0] < e2[1] <= e1[1]

def merge(f, j, wf=None):
	line_index = 0
	result = []
	for line in f.readlines():
		
		# print_flag = line.startswith("\" ")
		line = preprocess(line)
		relation = line.strip().split(",")[-1]
		plain_text = change_line(line).strip()
		target = None
		if plain_text == "": continue
		for item in j:
			# if item["text"].startswith("독일어"):
			# 	print(item["text"], len(item["text"])) 
			# 	print(plain_text, len(plain_text)) 
			# 	print(item["text"] == plain_text)
			if item["text"] == plain_text:
				target = item
				break
		else:
			continue
			# raise Exception("No such sentence: %s @ line %d" % (plain_text, line_index))
		line_index += 1
		line = ",".join(line.split(",")[:-1]).strip('"').strip()
		# line = line.strip('"').strip()
		remove_text = ["<e1>", "</e1>", "<e2>", "</e2>"]

		inds = []
		s=0
		e=0
		while True:
			try:
				s = line.index("<e1>", e)
				e = line.index("</e1>", e+5)
				inds.append([s, e, 1, line[s+4:e]])
				if line[s+4:e] in ["인천국제공항고속도로"]:
					print_flag = True
				# print(s,e, 1)
			except Exception:
				break
		s=0
		e=0
		while True:
			try:
				s = line.index("<e2>", e)
				e = line.index("</e2>", e+5)
				inds.append([s, e, 2, line[s+4:e]])
				# print(s,e, 2)
			except Exception:
				break
		# inds = list(map(lambda x: line.index(x), remove_text))
		inds = sorted(inds, key=lambda x: x[0])
		# print(inds)
		for i in range(len(inds)):
			inds[i][0] -= 4 * i + 5 * i
			inds[i][1] -= 4 * (i+1) + 5 * i

		# for s, e, _, _ in inds:
		# 	print(plain_text[s:e])

		# print(e1, e2)
		# print(plain_text[inds[0]:inds[1]], plain_text[inds[2]:inds[3]])

		# assert(e1 == plain_text[inds[0]:inds[1]] and e2 == plain_text[inds[2]:inds[3]]) # index check
		# indexes = [[inds[0], inds[1], 1, e1], [inds[2], inds[3], 2, e2]]
		for item in target["entities"]:
			if "entity" not in item or item["entity"] in ["NIL", "#UNK#"]: continue
			skip = False
			for s, e, _, _ in inds:
				if overlaps((s, e), (item["start"], item["end"])):
					skip = True
					break
			if skip: continue
			# if "entity" not in item:
			# 	print(item["surface"])
			inds.append([item["start"], item["end"], 0, item["entity"]])
		ins = [("[", "]"), ("<e1>", "</e1>"), ("<e2>", "</e2>")]
		inds = sorted(inds, key=lambda x: x[0])
		text = plain_text
		for ind in inds:

			original_en_len = ind[1] - ind[0]
			text = text[:ind[0]] + ins[ind[2]][0] + text[ind[0]:]
			
			# if print_flag: print(text, 0)
			for ind2 in inds:
				ind2[0] += len(ins[ind[2]][0])
				ind2[1] += len(ins[ind[2]][0])
			
			if ind[3] != "NIL":
				text = text[:ind[0]] + ind[3] + text[ind[1]:]
				# if print_flag: print(text, 1)
				for ind2 in inds: 
					ind2[0] += len(ind[3]) - original_en_len
					ind2[1] += len(ind[3]) - original_en_len
			text = text[:ind[1]] + ins[ind[2]][1] + text[ind[1]:]
			# if print_flag: print(text, 2)
			for ind2 in inds:
				ind2[0] += len(ins[ind[2]][1])
				ind2[1] += len(ins[ind[2]][1])

		if wf is not None:
			# wf.write('"'+text+"\","+relation+"\n")
			wf.write('"'+text+'"'+"\n")



if __name__ == '__main__':
	import json
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", type=str, choices=["make_plain", "merge"], required=True)
	parser.add_argument("--input", type=str, required=True)
	# parser.add_argument("--target_dir", type=str, required=True)
	args = parser.parse_args()
	if args.mode == "merge":
		print("Merge")
		with open(args.input, encoding="UTF8") as f, open("tta_merged.json", encoding="UTF8") as jf, open(args.input+"_result.csv", "w", encoding="UTF8") as wf:
			merge(f, json.load(jf), wf)
	elif args.mode == "make_plain":
		print("Make Plain")
		with open(args.input, encoding="UTF8") as f, open("tta_plain.txt", "w", encoding="UTF8") as wf:
			for item in change_file(f):
				if item == "": continue
				wf.write(item+"\n")