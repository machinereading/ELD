from src.utils import jsonload, readfile

# j = jsonload("runs/ec/StatisticClusteringResult.json")
j = jsonload("ec_result.json")
c = 0
for item in j:
	for word in item:
		c += 1
v = set([])
for line in readfile("corpus/de_set.set"):
	v.add(line.split(" ")[1][1:-1])

print(len(v))
print(c)