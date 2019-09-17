from .ContextEmbedder import *

class ERScorer(nn.Module):
	def __init__(self, args, input_dim):
		super(ERScorer, self).__init__()
		self.embedder = BiContextEmbedder(args, input_dim)
		self.scorer = nn.Sequential(
				nn.Dropout(),
				nn.Linear(args.er_output_dim * 2, 50),
				nn.Dropout(),
				nn.ReLU(),
				nn.Linear(50, 1),
				nn.Sigmoid()
		)

	def forward(self, lctx, rctx):
		embedding = self.embedder(lctx, rctx)
		score = self.scorer(embedding)
		return score

	def loss(self, prediction, label):
		return F.binary_cross_entropy(prediction, label)

class ELScorer(nn.Module):
	def __init__(self, args, input_dim):
		super(ELScorer, self).__init__()
		self.embedder = BiContextEmbedder(args, input_dim)
		self.scorer = nn.Sequential(
				nn.Dropout(),
				nn.Linear(args.el_output_dim * 2, 50),
				nn.Dropout(),
				nn.ReLU(),
				nn.Linear(50, 1),
				nn.Sigmoid()
		)

	def forward(self, lctx, rctx):
		embedding = self.embedder(lctx, rctx)
		score = self.scorer(embedding)
		return score

	def loss(self, prediction, label):
		return F.binary_cross_entropy(prediction, label)

class ECScorer(nn.Module):
	def __init__(self, args, er_scorer, el_scorer):
		super(ECScorer, self).__init__()
		self.wctx2emb = er_scorer.embedder
		self.ectx2emb = el_scorer.embedder

	def loss(self, prediction, label):
		return F.nll_loss(prediction, label)

	def forward(self, er_score, el_score, er_emb, el_emb):
		# [er_emb, el_emb] --> token embedding
		we = self.wctx2emb(er_emb)
		ee = self.ectx2emb(el_emb)
		cat = torch.cat([we, ee])
		cluster_avg_emb = cat.mean(1)  # how to ignore zero padding??

		# avg embedding shape = [batch size * (word embedding size + entity embedding size)]
		# make average embedding to [batch size * token size * (word embedding size + entity embedding size)]
		# return F.relu(er_score * ec_score * torch.exp(F.cosine_similarity(cat, cluster_avg_emb)))

class JointScorer(nn.Module):
	def __init__(self, args, er_input_dim, el_input_dim):
		super(JointScorer, self).__init__()
		self.er_embedder = BiContextEmbedder(args, er_input_dim)
		self.el_embedder = BiContextEmbedder(args, el_input_dim)
		self.scorer = nn.Sequential(
				nn.Dropout(),
				nn.Linear((er_input_dim + el_input_dim) * 2, 100),
				nn.ReLU(),
				nn.Dropout(),
				nn.Linear(100, 3),
				nn.Sigmoid()
		)

	def forward(self, er_lctx, er_rctx, el_lctx, el_rctx):
		er_emb = self.er_embedder(er_lctx, er_rctx)
		el_emb = self.el_embedder(el_lctx, el_rctx)
		return self.scorer(torch.cat([er_emb, el_emb], -1))

	def loss(self, pred, label):
		return F.cross_entropy(pred, label)


class EmbedScorer(nn.Module):
	def __init__(self, args, jamo_input_dim, w_input_dim, e_input_dim):
		super(EmbedScorer, self).__init__()
		self.device = args.device
		self.scorer = nn.Sequential(
				nn.Dropout(),
				nn.Linear(jamo_input_dim + (w_input_dim + e_input_dim) * 2, 100),
				nn.ReLU(),
				nn.Dropout(),
				nn.Linear(100, 1),
				nn.Sigmoid()
		)

	def forward(self, j, w, e):
		score = self.scorer(torch.cat((j, w, e), -1))
		ones = torch.ones(score.size()).to(self.device)
		zeros = torch.zeros(score.size()).to(self.device)
		return torch.where(score > 0.5, ones, zeros)
