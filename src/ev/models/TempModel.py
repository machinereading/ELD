import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils import readfile
from ... import GlobalValues as gl
from . import ModelFactory


class Model(nn.Module):
	def __init__(self, args):
		super(Model, self).__init__()
		self.word2idx = {x: i for i, x in enumerate(readfile(args.word_dict_path))}
		self.idx2word = {i: x for x, i in self.word2idx.items()}
		self.ent2idx = {x: i for i, x in enumerate(readfile(args.entity_dict_path))}
		self.idx2ent = {i: x for x, i in self.ent2idx.items()}
		self.word_embedding = nn.Embedding.from_pretrained(args.word_embedding_path)
		self.entity_embedding = nn.Embedding.from_pretrained(args.entity_embedding_path)
		self.er_model = ModelFactory.load_model(args.er_model, ).cuda()
		self.el_model  = ModelFactory.load_model().cuda()
		self.ec_model = ModelFactory.load_model().cuda()
		self.scorer = nn.Linear(3, 1)
		self.optimizer = torch.optim.Adam([self.er_model, self.ec_model, self.scorer])

	def forward(self, batch):
		er_score = 0.0
		el_score = 0.0
		ec_score = 0.0

		final_score = self.scorer(torch.FloatTensor([er_score, el_score, ec_score]))
		final_score = F.sigmoid(final_score)
		loss = 0