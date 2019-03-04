from src.utils import readfile, writefile

wf = open("corpus/er/cw_fix.conll", "w" , encoding="UTF8")

for line in readfile("corpus/er/crowdsourcing1.conll"):
	if len(line.strip()) == 0:
		wf.write("\n")
		continue
	i, w, d1, d2, tag = line.strip().split(" ")
	tag = tag.replace("/", "-")
	wf.write(" ".join([i, w, d1, d2, tag])+"\n")
wf.close()