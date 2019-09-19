import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .modules import SeparateEncoderBasedTransformer
from .utils import ELDArgs, DataModule, Evaluator
from .. import GlobalValues as gl
class ELDMain:
	def __init__(self, mode: str, model_name: str):
		assert mode in ["train", "eval", "demo"]
		if mode != "train":
			self.args = ELDArgs(model_name)
			self.transformer.load_state_dict(torch.load(self.args.model_path))
		else:
			try:
				self.args = ELDArgs.from_json("models/eld/%s_args.json" % model_name)
			except FileNotFoundError:
				gl.logger.critical("No argument file exists!")
		self.args.mode = mode
		self.data = DataModule(mode, self.args)
		self.evaluator = Evaluator(self.args, self.data)
		self.entity_index = {}
		self.i2e = {v: k for k, v in self.entity_index.items()}
		self.entity_embedding = torch.zeros().cuda()
		self.entity_embedding_dim = self.entity_embedding[0].size()[-1]
		self.map_threshold = 0.5

		self.transformer = SeparateEncoderBasedTransformer(self.args)



	def train(self):
		train_batch = DataLoader(dataset=self.data.train_dataset, batch_size=32, shuffle=True, num_workers=4)
		dev_batch = DataLoader(dataset=self.data.dev_dataset, batch_size=32, shuffle=False, num_workers=4)
		tqdmloop = tqdm(range(1, self.args.epochs + 1))
		optimizer = torch.optim.Adam(self.transformer.parameters(), lr=1e-4, weight_decay=1e-4)
		gold_corpus = self.data.dev_corpus
		max_score = 0
		max_score_epoch = 0

		for epoch in tqdmloop:
			self.transformer.train()
			for batch in train_batch:
				optimizer.zero_grad()
				ce, we, ee, re, te = [x.to("cuda") if x is not None else None for x in batch[:-1]]  # label 빼고
				label = batch[-1]
				pred = self.transformer(ce, we, ee, re, te)
				loss = self.transformer.loss(pred, label)
				loss.backward()
				optimizer.step()
				tqdmloop.set_description("Epoch %d, Loss %.4f" % (epoch, loss))
			if epoch % self.args.eval_per_epoch == 0:
				self.transformer.eval()
				for batch in dev_batch:
					ce, we, ee, re, te = [x.to("cuda") if x is not None else None for x in batch[:-1]]  # label 빼고
					pred = self.transformer(ce, we, ee, re, te)
					label = batch[-1]
					pred_corpus = self.data.postprocess(None, pred, make_copy=True)
					score = self.evaluator.evaluate(gold_corpus, pred_corpus)
					gl.logger.info("Epoch %d - Score %.4f" % (epoch, score))
					if score > max_score:
						max_score = score
						max_score_epoch = epoch
						torch.save(self.transformer.state_dict(), self.args.model_path)
					gl.logger.info("Best epoch %d - Score %.4f" % (max_score_epoch, max_score))

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
		register_form = "__" + surface.replace(" ", "_")
		self.entity_index[register_form] = idx
		self.i2e[idx] = register_form
		self.entity_embedding = torch.cat((self.entity_embedding, entity_embedding.unsqueeze(0)), 0)
		return register_form

	def __call__(self, data):
		return self.predict(data)
