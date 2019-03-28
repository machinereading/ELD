import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import ModelFactory

class ERScorer(nn.Module):
	def __init__(self, args, input_dim):
		super(ERScorer, self).__init__()
		self.model = ModelFactory.load_model(args.er_model, input_dim, args.er_output_dim)


	def forward(self, word_embeddings):
		return F.relu(self.model(word_embeddings))

	def loss(self, prediction, label):
		return F.cross_entropy(prediction, label)

class ELScorer(nn.Module):
	def __init__(self, args, input_dim):
		super(ELScorer, self).__init__()
		self.model = ModelFactory.load_model(args.el_model, input_dim, args.el_output_dim)

	def forward(self, entity_embeddings):
		return F.relu(self.model(entity_embeddings))
	
	def loss(self, prediction, label):
		return F.cross_entropy(prediction, label)

class ECScorer(nn.Module):
	def __init__(self, args):
		super(ECScorer, self).__init__()
		self.wctx2emb = None
		self.ectx2emb = None
	
	def loss(self, prediction, label):
		return F.cross_entropy(prediction, label)

	def forward(self, er_score, el_score, er_emb, el_emb):
		# [er_emb, el_emb] --> token embedding
		we = self.wctx2emb(er_emb)
		ee = self.ectx2emb(el_emb)
		cluster_size = ee.shape[1]
		cat = torch.cat([we, ee])
		cluster_avg_emb = cat.mean(1) # how to ignore zero padding??
		
		# avg embedding shape = [batch size * (word embedding size + entity embedding size)]
		# make average embedding to [batch size * token size * (word embedding size + entity embedding size)]
		return F.relu(er_score * ec_score * torch.exp(F.cosine_similarity(cat, cluster_avg_emb)))
