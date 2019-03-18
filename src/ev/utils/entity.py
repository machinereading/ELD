import torch.nn as nn
from ...utils import readfile
class EntityEmbedding(nn.Embedding):
	def __init__(self, args):
		self.entity_path = args.entity_path
		self.embedding_dim = args.entity_embedding_dimension
		entity_dict = [x for x in readfile(self.entity_path)]
		super(EntityEmbedding, self).__init__(len(entity_dict), self.embedding_dim)
		