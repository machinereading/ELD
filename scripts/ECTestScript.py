from src.ec.ECMain import EC

module = EC()
for cluster in module.cluster("corpus/test.set"):
	print(cluster)