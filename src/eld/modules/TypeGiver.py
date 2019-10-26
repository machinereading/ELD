import torch
import torch.nn as nn

from ..utils import ELDArgs

class TypeGiver(nn.Module):
	def __init__(self, in_dim, out_dim):
		super(TypeGiver, self).__init__()
		self.typegiver = nn.Linear(in_dim, out_dim)

	def forward(self, entity_encoding):
		pass
