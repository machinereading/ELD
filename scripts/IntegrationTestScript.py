import os
import json

from src.el.ELMain import EL
from src.ec.ECMain import EC
from src.utils import TimeUtil, jsonload, jsondump


source_dir = "corpus/el_golden_postprocessed_marked/"
target_dir = "runs/integration/"
source = []
for x in os.listdir(source_dir):
	d = jsonload(source_dir + x)
	d["fileName"] = x
	source.append(d)
print(len(source))
with TimeUtil.TimeChecker("EL"):
	el_module = EL("test", "old_candidates")
	el_result = el_module.predict(source, "CROWDSOURCING")
	for batch in el_result:
		for item in batch:
			jsondump(item, target_dir+item["fileName"])

TimeUtil.time_analysis()