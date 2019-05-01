import torch
from . import TimeUtil


@TimeUtil.measure_time
def nonzero_avg_stack(tensor):
	# input: 3 dimension tensor - max voca * max jamo * embedding size
	# output: 2 dimension tensor - max voca * embedding size
	# 지금 이거 원하는대로 동작 안함. 지금 임베딩이 빠지는 중이니까 확인 필요
	nz = nonzero_item_count(tensor)
	tensor = tensor.sum(1) / nz if nz > 0 else tensor
	return tensor


def nonzero_item_count(tensor):
	# input: 2 dimension tensor
	# output: single tensor
	result = 0
	for vec in tensor:
		if torch.sum(vec) == 0:
			continue
		result += 1
	return result