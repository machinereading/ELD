from src.utils import jsonload, jsondump
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, average_precision_score
import math
import matplotlib.pyplot as plt
models = ["discovery", "discovery_degree_surface", "discovery_surface", "discovery_degree"]
result = {m: {} for m in models}

def ms(l):
	avg = sum(l) / len(l)
	std = math.sqrt(sum([(x - avg) ** 2 for x in l]))
	return [avg, std]

for model in models:
	scores = {i: [] for i in range(1, 10)}
	max_score = 0
	max_preds = []

	for i in range(5):
		name = model + "_" + str(i)
		result_json = jsonload("runs/eld/%s/%s_test.json" % (name, name))
		label = [item["NewEntLabel"] for item in result_json["data"]]

		pred = [item["NewEntPred"] for item in result_json["data"]]
		# print(label, pred)
		for threshold in range(1, 10):
			p, r, f, _ = precision_recall_fscore_support(label, [1 if x > threshold * 0.1 else 0 for x in pred], average="binary")
			scores[threshold].append([p, r, f])
			if f > max_score:
				max_score = f
				max_preds = pred
	for k, v in scores.items():
		p = ms([x[0] for x in v])
		r = ms([x[1] for x in v])
		f = ms([x[2] for x in v])
		result[model][k] = [p, r, f]

	precision, recall, _ = precision_recall_curve(label, max_preds)
	average_precision = average_precision_score(label, max_preds)
	plt.step(recall, precision, color='b', alpha=0.2,
	         where='post')
	plt.fill_between(recall, precision, alpha=0.2, color='b')

	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.0])
	plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
	          average_precision))
	plt.savefig("%s.png" % model)
	# result[model] = {i: [list(ms(v[0])), list(ms(v[1])), list(ms(v[2]))] for i, v in scores.items()}

jsondump(result, "discovery_result.json")
