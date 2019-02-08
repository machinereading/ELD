from src.ec.ECMain import EC
from src.utils import TimeUtil

corpus = "corpus/train-cold.set"
test = "corpus/test.set"
with TimeUtil.TimeChecker("EC Init"):
	module = EC()

with TimeUtil.TimeChecker("EC Train"):
	module.train(corpus, test)
