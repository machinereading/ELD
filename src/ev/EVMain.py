from . import model_dict
from .models.TempModel import Model
from .utils.args import EV_Args
from .utils.data import DataGenerator

import torch
import torch.nn as nn
class EV():
	def __init__(self):
		# self.model = model_dict[model_name]
		self.args = EV_Args()
		self.data = DataGenerator(self.args)
		self.data.generate_data()
		self.model = Model(self.args)

	def train(self):
		pass

	def validate(self, entity_set):
		pass