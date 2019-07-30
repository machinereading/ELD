import torch
import torch.nn.functional as F
from tqdm import tqdm

from .models import Transformer
from .utils import ELDArgs

class ELDMain:
	def __init__(self):
		self.args = ELDArgs()
		self.entity_index = {}
		self.i2e = {v: k for k, v in self.entity_index.items()}
		self.entity_embedding = torch.zeros().cuda()
		self.map_threshold = 0.5

		self.transformer = Transformer()

	def train(self):
		tqdmloop = tqdm(range(1, self.args.epochs + 1))
		for epoch in tqdmloop:
			self.transformer.train()

			if epoch % self.args.eval_per_epoch == 0:
				self.transformer.eval()


	def predict(self, data, register=True):
		pred_embedding = self.transformer(data)
		pred_embedding.repeat(len(self.entity_index))
		cos_sim = F.cosine_similarity(pred_embedding, self.entity_embedding)
		if cos_sim.max() > self.map_threshold: return self.i2e[cos_sim.argmax(dim=-1).cpu().data]
		if register:
			register_form = self.register_new_entity(data, pred_embedding)
			return register_form
		return None

	def register_new_entity(self, surface, entity_embedding):
		idx = len(self.entity_index)
		register_form = "_" + surface
		self.entity_index[register_form] = idx
		self.i2e[idx] = register_form
		self.entity_embedding = torch.cat((self.entity_embedding, entity_embedding.unsqueeze(0)), 0)
		return register_form

	def __call__(self, data):
		return self.predict(data)
