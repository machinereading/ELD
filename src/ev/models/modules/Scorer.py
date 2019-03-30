import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import ModelFactory

class BiContextEREmbedder(nn.Module):
	def __init__(self, args, input_dim):
		super(BiContextEREmbedder, self).__init__()
		# self.lctx_model = ModelFactory.load_model(args.er_model, input_dim, args.er_output_dim)
		self.lctx_model = nn.LSTM(input_size=input_dim, hidden_size=args.er_output_dim, bidirectional=False)
		self.rctx_model = nn.LSTM(input_size=input_dim, hidden_size=args.er_output_dim, bidirectional=False)

	def forward(self, lctx, rctx):
		lctx_emb = self.lctx_model(lctx)
		rctx_emb = self.rctx_model(rctx)
		return F.relu(torch.cat([lctx_emb, rctx_emb], -1))

	

class ERScorer(nn.Module):
	def __init__(self, args, input_dim):
		super(ERScorer, self).__init__()
		self.embedder = BiContextEREmbedder(args, input_dim)
		self.scorer = nn.Sequential(
			nn.Dropout(),
			nn.Linear(input_dim, 50),
			nn.ReLU(),
			nn.Linear(50, 1),
			nn.Sigmoid()
		)
	def forward(self, lctx, rctx):
		embedding = self.embedder(lctx, rctx)
		score = self.scorer(embedding)
		return score

	def loss(self, prediction, label):
		return F.cross_entropy(prediction, label)
	

class BiContextELEmbedder(nn.Module):
	def __init__(self, args, input_dim):
		super(BiContextELEmbedder, self).__init__()
		self.lctx_model = nn.LSTM(input_size=input_dim, hidden_size=args.el_output_dim, bidirectional=False)
		self.rctx_model = nn.LSTM(input_size=input_dim, hidden_size=args.el_output_dim, bidirectional=False)

	def forward(self, lctx, rctx):
		lctx_emb = self.lctx_model(lctx)
		rctx_emb = self.rctx_model(rctx)
		return F.relu(torch.cat([lctx_emb, rctx_emb], -1))
	

class ELScorer(nn.Module):
	def __init__(self, args, input_dim):
		super(ELScorer, self).__init__()
		self.embedder = BiContextELEmbedder(args, input_dim)
		self.scorer = nn.Sequential(
			nn.Dropout(),
			nn.Linear(input_dim, 50),
			nn.ReLU(),
			nn.Linear(50, 1),
			nn.Sigmoid()
		)
	def forward(self, lctx, rctx):
		embedding = self.embedder(lctx, rctx)
		score = self.scorer(embedding)
		return score

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
