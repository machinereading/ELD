from .. import GlobalValues as gl
from ..ds import Corpus
from ..utils import jsondump
from .models.IntraClusterModel import ThreeScorerModel, JointScorerModel
from .models.ValidationModel import ValidationModel
from .utils.args import EVArgs
from .utils.data import DataModule, SentenceGenerator, ClusterGenerator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import sklearn.metrics as metrics

import logging

class EV():
	def __init__(self, mode, model_name, config_file=None):
		# initialize arguments
		self.mode = mode
		if mode == "train":
			self.args = EVArgs() if config_file is None else EVArgs.from_config(config_file)
		else:
			try:
				self.args = EVArgs.from_json("data/ev/%s_args.json" % model_name)
			except:
				gl.logger.info("No argument file exists!")
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
			gl.logger.info("Cluster size:", len(self.dataset.corpus.cluster_list))
			gl.logger.info("Cluster out of KB:", len([x for x in self.dataset.corpus.cluster_list if not x.is_in_kb]))
			self.sentence_train, self.sentence_dev = self.dataset.corpus.split_sentence_to_dev()
			self.cluster_train, self.cluster_dev = self.dataset.corpus.split_cluster_to_dev()
			self.args.max_jamo = self.dataset.corpus.max_jamo
			jsondump(self.args.to_json(), "data/ev/%s_args.json" % model_name)
		# load / initialize model
		
		self.validation_model = ValidationModel(self.args).to(self.args.device)
		
		try:
			gl.logger.info("Loading model from %s" % self.args.validation_model_path)
			self.validation_model.load_state_dict(torch.load(self.args.validation_model_path))
			gl.logger.info("Validation model loaded")
		except:
			if self.mode == "train":
				gl.logger.info("Creating new validation model")
			else:
				import traceback
				traceback.print_exc()
				raise Exception("Model %s not exists!" % model_name)
		gl.logger.debug("Total number of parameters: %d" % sum(p.numel() for p in self.validation_model.parameters() if p.requires_grad))

		# pretrain
		# pretrain is required to fix inner models of scorer, even if there is nothing to pretrain
		self.pretrain()
		

	def train(self):
		self.logger.info("Start EV Training")
		best_dev_f1 = 0
		best_dev_precision = 0
		best_dev_recall = 0
		best_epoch = 0
		optimizer = torch.optim.Adam(self.validation_model.parameters(), lr=self.args.lr)
		train_generator = ClusterGenerator(self.cluster_train, for_train=True)
		train_dataloader = DataLoader(train_generator, batch_size=self.batch_size, shuffle=True)
		dev_generator = ClusterGenerator(self.cluster_dev)
		dev_dataloader = DataLoader(dev_generator, batch_size=self.batch_size, shuffle=False)
		for epoch in tqdm(range(1, self.args.epoch+1), desc="Training..."):
			self.validation_model.train()
			tp = []
			tl = []
			for batch in train_dataloader:
				jamo, wl, wr, el, er, size, label = [x.to(self.args.device) for x in batch]
				# jamo.to(self.args.device), wl.to(self.args.device), wr.to(self.args.device), el.to(self.args.device), er.to(self.args.device), label.to(self.args.device)
				optimizer.zero_grad()
				pred = self.validation_model(jamo, wl, wr, el, er, size).view(-1)
				li = 0
				labels = []
				for s in [(x-1) // self.args.chunk_size + 1 for x in size]:
					for _ in range(s):
						labels.append(label[li])
					li += 1
				loss = self.validation_model.loss(pred, torch.FloatTensor(labels).to(self.args.device))
				loss.backward()
				optimizer.step()
				# tp += [1 if x > 0.5 else 0 for x in pred]
				# tl += [x.data for x in labels]
			# gl.logger.info(tp[:10], tl[:10])
			# gl.logger.info("Train F1: %.2f" % metrics.f1_score(tl, tp, labels=[0,1], average="micro"))
			if epoch % self.args.eval_per_epoch == 0:
				self.validation_model.eval()
				preds = []
				labels = []
				for batch in dev_dataloader:
					jamo, wl, wr, el, er, size, label = [x.to(self.args.device) for x in batch]
					pred = self.validation_model(jamo, wl, wr, el, er, size)
					pi = 0
					scores = []
					for s in [(x-1) // self.args.chunk_size + 1 for x in size]:
						scores.append(torch.mean(pred[pi:pi+s]))
						pi += s
					preds += [1 if x > 0.5 else 0 for x in scores]
					labels += list(label.to(device="cpu", dtype=torch.int32).numpy())

				f1 = metrics.f1_score(labels, preds, labels=[0,1], average="micro")
				gl.logger.info("F1: %.2f" % (f1 * 100))
				# import sys
				# sys.exit(0)
				if f1 > best_dev_f1:
					best_dev_f1 = f1
					best_epoch = epoch
					torch.save(self.validation_model.state_dict(), self.args.validation_model_path)
				gl.logger.info("Best F1: %.2f @ epoch %d" % (best_dev_f1 * 100, best_epoch))
	
	def pretrain(self):
		gl.logger.info("Start EV Pretraining")

		# self.cluster_model.pretrain(SentenceGenerator(self.sentence_train), SentenceGenerator(self.sentence_dev))
		# self.cluster_model.pretrain(self.dataset)
		# self.validation_model.pretrain(self.cluster_generator)
		gl.logger.info("Pretraining Done")

	def validate(self, corpus):
		# entity set to tensor
		assert type(corpus) is dict or type(corpus) is Corpus
		corpus = self.dataset.convert_cluster_to_tensor(corpus, max_jamo_restriction=self.args.max_jamo)
		loader = DataLoader(ClusterGenerator(corpus, filter_nik=True), batch_size=self.batch_size, shuffle=False)

		# validate tensor
		preds = []
		for jamo, wl, wr, el, er, size, _ in loader:
			jamo, wl, wr, el, er = jamo.to(self.args.device), wl.to(self.args.device), wr.to(self.args.device), el.to(self.args.device), er.to(self.args.device)
			pred = self.validation_model(jamo, wl, wr, el, er, size)
			pi = 0
			scores = []
			for s in [(x-1) // self.args.chunk_size + 1 for x in size]:
				scores.append(torch.mean(pred[pi:pi+s]))
				pi += s
			preds += [1 if x > 0.5 else 0 for x in scores]
		# mark
		for cluster, prediction in zip(corpus.cluster_list, preds):
			cluster.kb_uploadable = prediction > 0.5
		return corpus
		

	def __call__(self, corpus):
		return self.validate(corpus)

	@property
	def word_embedding(self):
		return self.validation_model.we

	@property
	def entity_embedding(self):
		return self.validation_model.ee
		