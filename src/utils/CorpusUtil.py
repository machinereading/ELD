# generate word, pos, label set from data
import json
import os
import itertools
from .. import GlobalValues as gl
from ..utils import progress, printfunc
def extract(generator):

	
	pfn = "pos.json"
	lfn = "label.json"
	# if os.path.isfile(gl.target_dir+pfn) and os.path.isfile(gl.target_dir+lfn):
	# 	with open(gl.target_dir+pfn, encoding="UTF8") as f:
	# 		pj = json.load(f)
	# 	with open(gl.target_dir+lfn, encoding="UTF8") as f:
	# 		lj = json.load(f)
	# 	return pj, lj
	# if gl.run_mode in ["eval", "predict"]: raise Exception("No initialized trainset")
	print("Extracting pos and label from corpus...")
	pos_tag_set = set([])
	label_set = set([])
	x = 0
	longest = 0
	for w, p, l in generator:
		for word in w:
			if len(word) > longest:
				longest = len(word)
		for label in l:
			label_set.add(label)
		for pos in p:
			pos_tag_set.add(pos)
		x += 1
		if x % 1000 == 0:
			printfunc(str(x))
	printfunc(str(x))
	print()
	gl.train_set_count = x
	gl.longest_word_len = longest
	pj = {}
	lj = {}
	with open(gl.target_dir+pfn, "w", encoding="UTF8") as f:
		c = 0
		for pos in pos_tag_set:
			pj[pos] = c
			c += 1
		json.dump(pj, f, ensure_ascii=False, indent="\t")
	with open(gl.target_dir+lfn, "w", encoding="UTF8") as f:
		c = 0
		for label in label_set:
			lj[label] = c
			c += 1
		json.dump(lj, f, ensure_ascii=False, indent="\t")
	print("Done!")
	return pj, lj