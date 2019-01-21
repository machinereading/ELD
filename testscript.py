
# from konlpy.tag import Okt
test_text1 = "지미 카터는 미국의 제 37대 대통령이다."
test_text2 = ""




# okt = Okt()
# morphs = okt.morphs(test_text)
# print("\n".join(morphs))
from src.el.ELMain import EL
from src.utils import TimeUtil
import os
import json

# os.environ["CUDA_VISIBLE_DEVICES"]="1"
if __name__ == '__main__':
	
	module = EL()
	with open("errors.txt", encoding="UTF8") as f, open("error_parse_result.json", "w", encoding="UTF8") as wf:
		json.dump(module(list(filter(lambda x: len(x.strip()) > 0, f.readlines()))), wf, ensure_ascii=False, indent="\t")
	TimeUtil.time_analysis()