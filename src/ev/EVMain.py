from ... import GlobalValues as gl
from . import model_dict
from .models.Model import Model
from .utils.args import EVModelArgs, EVDataArgs
from .utils.data import DataGenerator

import logging

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class EV():
	def __init__(self):
		# initialize arguments
		self.model_args = EVModelArgs()
		self.data_args = EVDataArgs()
		
		# load / initialize model
		self.model = Model(self.args)
		if torch.cuda.is_available():
			self.model.cuda()

		
		# load / generate data
		self.data = DataGenerator(self.args)
		

		

	def train(self):
		logging.info("Start EV Training")
		best_dev_f1 = 0
		best_dev_precision = 0
		best_dev_recall = 0
		best_epoch = 0
		optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
		self.model.train()
		for epoch in tqdm(range(self.args.epoch), desc="Training..."):
			for batch in self.data.get_tensor_batch():
				optimizer.zero_grad()
				loss = self.model.loss(batch)
				loss.backward()
				optimizer.step()

	def validate(self, entity_set):
		pass

	def __call__(self, entity_set):
		return self.validate(entity_set)