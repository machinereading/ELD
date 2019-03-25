import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils import readfile
from ... import GlobalValues as gl
from . import ModelFactory

class ValidationModel(nn.Module):
	def __init__(self, args):
		super(ValidationModel, self).__init__()

	def forward(self, kb, batch):
		pass