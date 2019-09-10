import sklearn.metrics as metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .models import ValidationModel, JointScorerModel
from .utils.args import EVArgs
from .utils.data import DataModule, ClusterGenerator
from .. import GlobalValues as gl
from ..ds import FakeCluster
from ..utils import jsondump
import traceback

class EV:
	def __init__(self, mode, model_name, config_file=None):
		# initialize arguments
		self.mode = mode
		if mode == "train":
			self.args = EVArgs(model_name) if config_file is None else EVArgs.from_config(config_file)
		else:
			try:
				self.args = EVArgs.from_json("models/ev/%s_args.json" % model_name)
			except FileNotFoundError:
				gl.logger.critical("No argument file exists!")
			except:
				gl.logger.critical("Error on loading argument file")
				import traceback
				traceback.print_exc()
		# self.args = EVArgs() if config_file is None else EVArgs.from_config(config_file)

		if torch.cuda.is_available():
			self.args.device = "cuda"
		else:
			self.args.device = "cpu"
		self.args.model_name = model_name
		self.batch_size = self.args.batch_size

		# load / generate data

		self.dataset = DataModule(mode, self.args)
		if self.mode == "train":
			gl.logger.info("Cluster size: %d" % len(self.dataset.corpus.cluster_list))
			gl.logger.info(
					"Cluster out of KB: %d" % len(
							[x for x in self.dataset.corpus.cluster_list if type(x) is FakeCluster]))
			self.sentence_train, self.sentence_dev = self.dataset.corpus.split_sentence_to_dev()
			self.cluster_train, self.cluster_dev = self.dataset.corpus.split_cluster_to_dev()
			self.args.max_jamo = self.dataset.corpus.max_jamo
			jsondump(self.args.to_json(), "models/ev/%s_args.json" % model_name)
		# load / initialize model
		if self.args.use_intracluster_scoring:
			self.intra_cluster_model = JointScorerModel(self.args).to(self.args.device)
		self.validation_model = ValidationModel(self.args).to(self.args.device)
		self.validation_model_parallel = nn.DataParallel(self.validation_model)
		try:
			gl.logger.info("Loading model from %s" % self.args.validation_model_path)
			if mode == "demo":
				map_location = lambda storage, loc: storage
				self.validation_model.load_state_dict(torch.load(self.args.validation_model_path, map_location=map_location))
			else:
				self.validation_model.load_state_dict(torch.load(self.args.validation_model_path))
			if self.args.use_intracluster_scoring:
				self.intra_cluster_model.load_state_dict(torch.load(self.args.joint_model_path))
			gl.logger.info("Validation model loaded")
			self.args.device = torch.device(self.args.device)
		except Exception:
			if self.mode == "train":
				gl.logger.info("Creating new validation model")
			else:
				import traceback
				traceback.print_exc()
				raise Exception("Model %s not exists!" % model_name)
		gl.logger.debug("Total number of parameters: %d" % sum(
				p.numel() for p in self.validation_model.parameters() if p.requires_grad))

	def train(self):
		gl.logger.info("Start EV Training")
		best_dev_f1 = 0
		best_epoch = 0
		optimizer = torch.optim.Adam(self.validation_model.parameters(), lr=self.args.lr)
		train_generator = ClusterGenerator(self.cluster_train, for_train=True)
		train_dataloader = DataLoader(train_generator, batch_size=self.batch_size, shuffle=True, pin_memory=True)
		dev_generator = ClusterGenerator(self.cluster_dev)
		dev_dataloader = DataLoader(dev_generator, batch_size=self.batch_size, shuffle=False, pin_memory=True)
		for epoch in range(1, self.args.epoch + 1):
			self.validation_model.train()
			tp = []
			tl = []
			err_count = 0
			for batch in train_dataloader:
				try:
					jamo, wl, wr, el, er, size, label = [x.to(self.args.device, non_blocking=True) for x in batch]
					# print(size)
					optimizer.zero_grad()

					pred = self.validation_model(jamo, wl, wr, el, er, size).view(-1)
					li = 0
					labels = []
					for s in [(x - 1) // self.args.chunk_size + 1 for x in size]:
						for _ in range(s):
							labels.append(label[li])
						li += 1
					# print(len(labels), len(pred))
					for l, p in zip(label, pred):
						print(l, p)
					loss = self.validation_model.loss(pred, torch.FloatTensor(labels).to(self.args.device))
					# loss = F.binary_cross_entropy(pred, torch.FloatTensor(labels).to(self.args.device))
					loss.backward()
					optimizer.step()
				# tp += [1 if x > 0.5 else 0 for x in pred]
				# tl += [x.data for x in labels]
				except:
					err_count += 1
					if err_count > 10:
						import traceback
						traceback.print_exc()

			if err_count > 0:
				gl.logger.debug("Epoch %d error %d" % (epoch, err_count))
			# gl.logger.info(tp[:10], tl[:10])
			# gl.logger.info("Train F1: %.2f" % metrics.f1_score(tl, tp, labels=[0,1], average="micro"))
			gl.logger.info("Epoch %d loss %f" % (epoch, loss))
			if epoch % self.args.eval_per_epoch == 0:
				self.validation_model.eval()
				preds = []
				labels = []
				for batch in dev_dataloader:
					jamo, wl, wr, el, er, size, label = [x.to(self.args.device, non_blocking=True) for x in batch]
					pred = self.validation_model(jamo, wl, wr, el, er, size)
					pi = 0
					scores = []
					for s in [(x - 1) // self.args.chunk_size + 1 for x in size]:
						scores.append(torch.mean(pred[pi:pi + s]))
						pi += s
					preds += [1 if x > 0.5 else 0 for x in scores]
					labels += list(label.to(device="cpu", dtype=torch.int32).numpy())

				f1 = metrics.f1_score(labels, preds, labels=[0, 1], average="micro")
				gl.logger.info("F1: %.2f" % (f1 * 100))
				# import sys
				# sys.exit(0)
				if f1 > best_dev_f1:
					best_dev_f1 = f1
					best_epoch = epoch
					torch.save(self.validation_model.state_dict(), self.args.validation_model_path)
				gl.logger.info("Best F1: %.2f @ epoch %d" % (best_dev_f1 * 100, best_epoch))
		gl.logger.info("Model %s train complete" % self.args.model_name)

	def validate(self, corpus):
		# entity set to tensor
		# assert type(corpus) is list or type(corpus) is Corpus


		corpus = self.dataset.convert_cluster_to_tensor(corpus, max_jamo_restriction=self.args.max_jamo)
		batch_size = 4
		loader = DataLoader(ClusterGenerator(corpus, filter_nik=True), batch_size=self.batch_size, pin_memory=True)
		gl.logger.info("Clusters to validate: %d" % len(corpus.cluster_list))
		# validate tensor
		preds = []
		error_count = 0
		error_print_flag = False
		# for item in corpus:
		self.validation_model.eval()
		with torch.no_grad():
			for batch in loader:
				try:
					jamo, wl, wr, el, er, size, _ = [x.to(self.args.device, non_blocking=True) for x in batch]
					# print(x, *[x.size() for x in batch])
					# jamo, wl, wr, el, er = jamo.to(self.args.device), wl.to(self.args.device), wr.to(self.args.device), el.to(
					# 		self.args.device), er.to(self.args.device)
					pred = self.validation_model_parallel(jamo, wl, wr, el, er, size).cpu()
					pi = 0
					scores = []
					for s in [(x - 1) // self.args.chunk_size + 1 for x in size]:
						scores.append(torch.mean(pred[pi:pi + s]))
						pi += s
					preds += scores
				except:
					if not error_print_flag:
						traceback.print_exc()
						error_print_flag = True
					preds += [0 for _ in range(self.batch_size)]
					error_count += self.batch_size
		# mark
		for cluster, prediction in zip(corpus.cluster_list, preds):
			cluster.upload_confidence = prediction
			cluster.kb_uploadable = bool(prediction > 0.5)
		gl.logger.info("Error count: %d" % error_count)
		return corpus

	def __call__(self, corpus):
		return self.validate(corpus)

class EVAll:
	def __init__(self):
		self.args = EVArgs("EVAll")
		self.args.device = "cuda"
		self.dataset = DataModule("test", self.args)

	def __call__(self, corpus):
		corpus = self.dataset.generate_fake_cluster(corpus)
		corpus = self.dataset.convert_cluster_to_tensor(corpus, max_jamo_restriction=108)
		for cluster in corpus.cluster_list:
			cluster.kb_uploadable = True
		return corpus

class EVRandom:
	def __init__(self):
		self.args = EVArgs("EVRandom")
		self.args.device = "cuda"
		self.dataset = DataModule("test", self.args)


	def __call__(self, corpus):
		corpus = self.dataset.generate_fake_cluster(corpus)
		corpus = self.dataset.convert_cluster_to_tensor(corpus, max_jamo_restriction=108)
		import random
		for cluster in corpus.cluster_list:
			cluster.kb_uploadable = random.random() > 0.5
		return corpus

class EVNone:
	def __init__(self):
		self.args = EVArgs("EVNone")
		self.args.device = "cuda"
		self.dataset = DataModule("test", self.args)

	def __call__(self, corpus):
		corpus = self.dataset.generate_fake_cluster(corpus)
		corpus = self.dataset.convert_cluster_to_tensor(corpus, max_jamo_restriction=108)
		for cluster in corpus.cluster_list:
			cluster.kb_uploadable = False
		return corpus
