import torch

def nonzero_std_dev(tensor):
	nzt = [torch.mean(torch.std(t[:nonzero_item_count(t)], dim=1)) for t in tensor]
	return torch.stack(nzt)

def nonzero_avg_stack(tensor):
	# input: 3 dimension tensor - max voca * max jamo * embedding size
	# output: 2 dimension tensor - max voca * embedding size
	nz = torch.tensor([nonzero_item_count(t) for t in tensor]).to(tensor.device, dtype=torch.float32).view(-1, 1) + 0.01
	tensor = tensor.sum(1) / nz
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
