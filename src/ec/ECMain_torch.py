import torch
import torch.nn as nn
from torch.utils import data
from tqdm import tqdm

from .CRTorch import CorefModel
from .utils import ECArgs
from .utils.data_e2etorch import DataModule, CorefDataset
from .. import GlobalValues as gl
from ..ds import Corpus
from ..utils import jsonload, jsondump

class EC:
	def __init__(self, mode, model_name):
		gl.logger.info("Initializing EC Model")
		self.mode = mode
		if mode == "train":
			self.args = ECArgs(model_name)
		else:
			try:
				self.args = ECArgs.from_json("models/ec/%s_args.json" % model_name)
			except FileNotFoundError:
				gl.logger.critical("No argument file exists!")
			except:
				gl.logger.critical("Error on loading argument file")
				import traceback
				traceback.print_exc()

		if torch.cuda.is_available():
			self.args.device = "cuda"
		else:
			self.args.device = "cpu"
		self.model = CorefModel(self.args)
		try:
			gl.logger.info("Loading model from %s" % self.args.model_path)
			self.model.load_state_dict(torch.load(self.args.model_path))
			gl.logger.info("Validation model loaded")
		except Exception:
			if self.mode == "train":
				gl.logger.info("Creating new clustering model")
				jsondump(self.args.to_json(), "models/ec/%s_args.json" % model_name)
			else:
				import traceback
				traceback.print_exc()
				raise Exception("Model %s not exists!" % model_name)
		self.dataset = DataModule(self.args)
		# self.model_parallel = nn.DataParallel(self.model)

	def train(self):
		import os
		dataset = [jsonload(self.args.data_path + fname) for fname in os.listdir(self.args.data_path)]
		corpus = self.dataset.prepare(Corpus.load_corpus(dataset))
		train_corpus, dev_corpus = corpus.split_sentence_to_dev()
		train_dataloader = data.DataLoader(CorefDataset(train_corpus), batch_size=self.args.batch_size, shuffle=True)
		dev_dataloader = data.DataLoader(CorefDataset(dev_corpus), batch_size=self.args.batch_size)
		optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
		best_f1 = 0
		best_epoch = 0
		avg_loss = 0
		for epoch in tqdm(range(self.args.epoch), desc="Training..."):
			self.model.train()
			for batch in train_dataloader:
				word_tensor, index, cluster_info, entity_len, precedent_label = [x.to(self.args.device) for x in batch]
				optimizer.zero_grad()
				prediction = self.model(word_tensor, index)
				loss = self.model.loss_v2(prediction, precedent_label)
				loss.backward()
				optimizer.step()
				avg_loss += loss
			if epoch % self.args.eval_epoch == 0:
				self.model.eval()
				tp, fp, fn, tn = 0, 0, 0, 0
				for batch in dev_dataloader:
					word_tensor, index, cluster_info, entity_len, precedent_label = [x.to(self.args.device) for x in batch]
					precedent_size = self.args.max_precedent + 1
					precedent = self.model(word_tensor, index)
					precedent.view(-1, precedent_size)
					precedent_label.view(-1, precedent_size)
					pred_index = torch.argmax(precedent, dim=1)
					for p, l in zip(pred_index, precedent_label):
						for pp in p:
							if pp == precedent_size - 1:
								if pp in l:
									tn += 1
								else:
									fn += 1
							else:
								if pp in l:
									tp += 1
								else:
									fp += 1

				p = tp / (tp + fp)
				r = tp / (tp + fn)
				f1 = 2 * p * r / (p + r)
				gl.logger.info("Average loss: %.4f" % (avg_loss / self.args.eval_epoch))
				gl.logger.info("F1: %.2f" % (f1 * 100))
				avg_loss = 0
				if best_f1 < f1:
					best_f1 = f1
					best_epoch = epoch
					torch.save(self.model.state_dict(), self.args.model_path)
				gl.logger.info("Best F1: %.2f @ epoch %d" % (best_f1 * 100, best_epoch))

	def cluster(self, input_data):
		corpus = self.dataset.prepare(input_data)
		print(type(corpus))
		loader = data.DataLoader(CorefDataset(corpus), batch_size=self.args.batch_size)
		clusters = []
		for batch in loader:
			word_tensor, index, _, entity_len, _ = [x.to(self.args.device) for x in batch]
			# batch_size = word_tensor.size()[0]
			prediction = self.model(word_tensor, index, entity_len)
			cluster = self.precedent_to_cluster(prediction, entity_len)
			clusters += cluster
		for s, c in zip(corpus, clusters):
			for t, tcinfo in zip(s.entities, c):
				t.cluster = tcinfo
		return corpus

	def __call__(self, input_data):
		return self.cluster(input_data)

	def indicator_to_cluster_prediction(self, prediction, indicator, entity_len, cluster_id=None):
		batch_size = prediction.size()[1]
		cluster_id_modify_flag = False
		if cluster_id is None:
			cluster_id_modify_flag = True
			cluster_id = [list(range(i)) for i in entity_len]
		prediction_ent = [
			[
				[-1 for _ in range(entity_len[i])]
				 for _ in range(entity_len[i])
			]
			for i in range(batch_size)]  # batch * entity * entity
		for p, (si, ei) in zip(prediction, indicator):
			for idx, pi in enumerate(p):  # 각각의 prediction, indicator
				if si == -1:
					si = ei
				if ei >= len(prediction_ent[idx]): continue
				prediction_ent[idx][ei][si] = pi
		# get max idx
		cluster_result = []
		for p, c in zip(prediction_ent, cluster_id):
			precede_list = []
			for i, ent_prob in enumerate(p):
				max_idx = ent_prob.index(max(ent_prob))
				precede_list.append(c[max_idx])
				if cluster_id_modify_flag:
					c[i] = c[max_idx]
			cluster_result.append(precede_list)
		return cluster_result

	def precedent_to_cluster(self, precedent, size):
		"""

		:param precedent: tensor - batch * max token size * (max precedent size + 1)
		:param size: tensor - batch(size info)
		:return: batch * token size(cluster id)
		"""
		precedent = torch.argmax(precedent)
		result = [[i for i in range(s)] for s in size]
		for idx, (item, s) in zip(precedent, size):
			for i, prec in enumerate(item):
				if i >= s: break
				if prec == self.args.max_precedent:
					target = i
				else:
					target = max(i - self.args.max_precedent, 0) + prec
				result[idx][i] = result[idx][target]
		return result
