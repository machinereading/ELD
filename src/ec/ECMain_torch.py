import torch
from torch.utils import data
from tqdm import tqdm

from .CRTorch import CorefModel
from .utils import ECArgs
from .utils.data_e2etorch import DataModule, CorefDataset
from .. import GlobalValues as gl
from ..ds import Corpus
from ..utils import jsonload
class EC:
	def __init__(self, mode, model_name):
		gl.logger.info("Initializing EC Model")
		self.mode = mode
		if mode == "train":
			self.args = ECArgs(model_name)
		else:
			try:
				self.args = ECArgs.from_json("models/ev/%s_args.json" % model_name)
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
			else:
				import traceback
				traceback.print_exc()
				raise Exception("Model %s not exists!" % model_name)
		self.dataset = DataModule(self.args)

	def train(self):
		import os
		dataset = [jsonload(self.args.data_path+fname) for fname in os.listdir(self.args.data_path)]
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
				word_tensor, index, cluster_info = [x.to(self.args.device) for x in batch]
				optimizer.zero_grad()
				prediction, indicator = self.model(word_tensor, index)
				loss = self.model.loss(prediction, indicator, cluster_info)
				loss.backward()
				optimizer.step()
				avg_loss += loss
			if (epoch + 1) % self.args.eval_epoch == 0:
				self.model.eval()
				tp, fp, fn, tn = 0, 0, 0, 0
				for batch in dev_dataloader:
					word_tensor, index, cluster_info = [x.to(self.args.device) for x in batch]
					prediction, indicator = self.model(word_tensor, index)
					real_labels = self.model.get_real_labels(indicator, cluster_info)
					cluster_info = cluster_info.transpose(0, 1)
					for pred, ind, label in zip(prediction, indicator, real_labels):
						pred = [0 if x < 0.5 else 1 for x in pred]
						mask = cluster_info[ind[1], :]
						for p, l, m in zip(pred, label, mask):
							if m != -1:
								if p > 0.5:
									if l == 1:
										tp += 1
									else:
										fp += 1
								else:
									if l == 1:
										fn += 1
									else:
										tn += 1
				p = tp / (tp+fp)
				r = tp / (tp+fn)
				f1 = 2*p*r/(p+r)
				gl.logger.info("Average loss: %.4f" % (avg_loss / self.args.eval_epoch))
				gl.logger.info("F1: %.2f" % (f1 * 100))
				avg_loss = 0
				if best_f1 < f1:
					best_f1 = f1
					best_epoch = epoch
					torch.save(self.model.state_dict(), self.args.model_path)
				gl.logger.info("Best F1: %.2f @ epoch %d" % (best_f1 * 100, best_epoch))


	def cluster(self, data):
		pass

	def __call__(self, data):
		return self.cluster(data)
