import numpy as np
from scipy import spatial
def cossim(a, b):
	return 1 - spatial.distance.cosine(a, b)


a = np.load("eld_test/mse_pred_cand_only_with_neg_softmax_without_jamo_4_cache_emb_testset.npy")

for item in a:
	for item2 in a:
		if np.sum(item - item2) == 0: continue
		print(cossim(item, item2))
	print()
