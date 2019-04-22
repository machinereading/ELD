from .. import GlobalValues as gl
from .models.IntraClusterModel import ThreeScorerModel, JointScorerModel
from .models.ValidationModel import ValidationModel
from .utils.args import EVArgs
from .utils.data import DataGenerator, SentenceGenerator, ClusterGenerator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import sklearn.metrics as metrics

class EV():
	def __init__(self, mode, model_name, config_file=None):
		# initialize arguments
		self.mode = mode
		self.args = EVArgs() if config_file is None else EVArgs.from_config(config_file)
		if torch.cuda.is_available():
			self.args.device = torch.device("cuda")
		else:
			self.args.device = torch.device("cpu")
		print(self.args.device)
		self.args.model_name = model_name
		self.batch_size = self.args.batch_size

		# load / generate data
		self.dataset = DataGenerator(mode, self.args)
		self.sentence_train, self.sentence_dev = self.dataset.corpus.split_sentence_to_dev()
		self.cluster_train, self.cluster_dev = self.dataset.corpus.split_cluster_to_dev()
		self.args.max_jamo = self.dataset.corpus.max_jamo
		# load / initialize model
		# self.cluster_model = JointScorerModel(self.args).to(self.args.device)
		# self.cluster_model = ThreeScorerModel(self.args).to(self.args.device)
		self.validation_model = ValidationModel(self.args).to(self.args.device)
		# print(sum(p.numel() for p in self.validation_model.parameters() if p.requires_grad))
		# import sys
		# sys.exit(0)
		# self.cluster_generator = ClusterGenerator(self.dataset.corpus)

		# pretrain
		# pretrain is required to fix inner models of scorer, even if there is nothing to pretrain
		self.pretrain()
		

	def train(self):
		gl.logger.info("Start EV Training")
		best_dev_f1 = 0
		best_dev_precision = 0
		best_dev_recall = 0
		best_epoch = 0
		optimizer = torch.optim.Adam(self.validation_model.parameters(), lr=self.args.lr)
		train_generator = ClusterGenerator(self.cluster_train)
		train_dataloader = DataLoader(train_generator, batch_size=self.batch_size, shuffle=True)
		dev_generator = ClusterGenerator(self.cluster_dev)
		dev_dataloader = DataLoader(dev_generator, batch_size=self.batch_size, shuffle=True)
		for epoch in tqdm(range(1, self.args.epoch+1), desc="Training..."):
			self.validation_model.train()
			for jamo, wl, wr, el, er, label in train_dataloader:
				jamo, wl, wr, el, er, label = jamo.to(self.args.device), wl.to(self.args.device), wr.to(self.args.device), el.to(self.args.device), er.to(self.args.device), label.to(self.args.device)
				optimizer.zero_grad()
				pred = self.validation_model(jamo, wl, wr, el, er)
				loss = self.validation_model.loss(pred, label.float())
				loss.backward()
				optimizer.step()
			if epoch % self.args.eval_per_epoch == 0:
				self.validation_model.eval()
				pred = []
				labels = []
				for jamo, wl, wr, el, er, label in train_dataloader:
					jamo, wl, wr, el, er, label = jamo.to(self.args.device), wl.to(self.args.device), wr.to(self.args.device), el.to(self.args.device), er.to(self.args.device), label.to(self.args.device)
					pred += [1 if x > 0.5 else 0 for x in self.validation_model(jamo, wl, wr, el, er)]
					labels += label
				f1 = metrics.f1_score(labels, pred)
				if f1 > best_dev_f1:
					best_dev_f1 = f1
					torch.save(self.validation_model.state_dict, self.args.validation_model_path)
	def pretrain(self):
		gl.logger.info("Start EV Pretraining")

		# self.cluster_model.pretrain(SentenceGenerator(self.sentence_train), SentenceGenerator(self.sentence_dev))
		# self.cluster_model.pretrain(self.dataset)
		# self.validation_model.pretrain(self.cluster_generator)
		gl.logger.info("Pretraining Done")

	def validate(self, corpus):
		# entity set to tensor
		assert type(corpus) is dict
		corpus = self.dataset.convert_cluster_to_tensor(corpus)
		loader = DataLoader(ClusterGenerator(corpus), batch_size=self.batch_size, shuffle=False)

		# validate tensor
		pred = []
		for jamo, wl, wr, el, er, _ in train_dataloader:
			jamo, wl, wr, el, er = jamo.to(self.args.device), wl.to(self.args.device), wr.to(self.args.device), el.to(self.args.device), er.to(self.args.device)
			pred += [1 if x > 0.5 else 0 for x in self.validation_model(jamo, wl, wr, el, er)]
			labels += label
		# mark
		for cluster, prediction in zip(cluster.entity_list, pred):
			cluster.kb_uploadable = prediction > 0.5
		

	def __call__(self, entity_set):
		return self.validate(entity_set)