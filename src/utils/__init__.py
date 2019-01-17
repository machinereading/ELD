import json
def progress(curprog, total, size=10):
	if curprog < 0 or total < 0 or size < 0:
		return ""
	a = int((curprog / total) * size)
	return "["+("*"*a)+(" "*(size - a)) + "]"

def printfunc(s):
	print("\r"+s, end="", flush=True)

def jsondump(obj, fname):
	with open(fname, "w", encoding="UTF8") as f:
		json.dump(obj, f, ensure_ascii=False, indent="\t")

def write(iterable, fname):
	with open(fname, "w", encoding="UTF8") as f:
		for item in iterable:
			f.write(item+"\n")

#useful macros
inv_dict = lambda x: {v: k for k, v in x.items()}
