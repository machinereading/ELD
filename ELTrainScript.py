from src.el.ELMain import EL
from src.utils import TimeUtil
from src.el.utils.eval import eval
import os
import random
import json
train_data_dir = "corpus/crowdsourcing_processed/"
test_data_dir = "corpus/el_golden_postprocessed_marked/"

module = EL("test", "new_candidates2")
train_set = []
dev_set = []
test_set = []
c = 0
for item in os.listdir(train_data_dir):
	with open(train_data_dir+item, encoding="UTF8") as f:
		if c % 10 != 0:
			train_set.append(json.load(f))
		else:
			dev_set.append(json.load(f))
	c += 1

# for item in os.listdir(test_data_dir):
# 	with open(test_data_dir+item, encoding="UTF8") as f:
# 		j = json.load(f)
# 		test_set.append(j)
# try:
# 	module.train(train_set, dev_set)
# except:
# 	import traceback
# 	traceback.print_exc()
eval(module, test_data_dir)
TimeUtil.time_analysis()