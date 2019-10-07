import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
	def __init__(self, in_dim):
		super(Classifier, self).__init__()
		self.weight = nn.Linear(in_dim, 1)

	def forward(self, tensor):
		return F.relu(self.weight(tensor))
