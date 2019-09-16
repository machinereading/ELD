import torch
import torch.nn.functional as F
from tqdm import tqdm

from .modules import Transformer
from .utils import ELDArgs

class ELDMain:
	def __init__(self):
		self.args = ELDArgs()
		self.entity_index = {}
		self.i2e = {v: k for k, v in self.entity_index.items()}
		self.entity_embedding = torch.zeros().cuda()
		self.entity_embedding_dim = self.entity_embedding[0].size()[-1]
		self.map_threshold = 0.5

		self.transformer = Transformer(self.args)

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
		if cos_sim.max() > self.map_threshold:
			target_ind = cos_sim.argmax(dim=-1).cpu().data
			target = self.i2e[target_ind]
			if self.args.modify_entity_embedding:
				pred_embedding *= self.args.modify_entity_embedding_weight
				self.entity_embedding += torch.stack([torch.zeros(self.entity_embedding_dim) for _ in range(target_ind - 1)] + [pred_embedding] + [torch.zeros(self.entity_embedding_dim) for _ in range(len(self.i2e) - target_ind - 1)])
			return target
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
