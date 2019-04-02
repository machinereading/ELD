from .. import GlobalValues as gl
from .models.IntraClusterModel import ThreeScorerModel, JointScorerModel
from .models.ValidationModel import ValidationModel
from .utils.args import EVArgs
from .utils.data import DataGenerator, SentenceGenerator, ClusterGenerator

import logging

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class EV():
	def __init__(self, model_name, config_file=None):
		# initialize arguments
		self.args = EVArgs() if config_file is None else EVArgs.from_config(config_file)
		if torch.cuda.is_available():
			self.args.device = torch.device("cuda")
		else:
			self.args.device = torch.device("cpu")
		self.args.model_name = model_name
		
		# load / initialize model
		self.cluster_model = JointScorerModel(self.args).to(self.args.device)
		self.validation_model = ValidationModel(self.args).to(self.args.device)
		
		# load / generate data
		dataset = DataGenerator(self.args)
		self.sentence_train, self.sentence_dev = dataset.corpus.split_sentence_to_dev()
		self.cluster_generator = ClusterGenerator(dataset.corpus)

		# pretrain
		# pretrain is required to fix inner models of scorer, even if there is nothing to pretrain
		self.pretrain()
		

	def train(self):
		logging.info("Start EV Training")
		best_dev_f1 = 0
		best_dev_precision = 0
		best_dev_recall = 0
		best_epoch = 0
		optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
		self.cluster_model.train()
		for epoch in tqdm(range(self.args.epoch), desc="Training..."):
			for batch in self.data.get_tensor_batch():
				optimizer.zero_grad()
				loss = self.cluster_model.loss(batch)
				loss.backward()
				optimizer.step()

	def pretrain(self):
		logging.info("Start EV Pretraining")

		self.cluster_model.pretrain(SentenceGenerator(self.sentence_train), SentenceGenerator(self.sentence_dev))
		self.validation_model.pretrain(self.cluster_generator)

	def validate(self, entity_set):
		pass

	def __call__(self, entity_set):
		return self.validate(entity_set)