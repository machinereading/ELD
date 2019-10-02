import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .modules.Transformer import SeparateEncoderBasedTransformer, JointTransformer
from .utils import ELDArgs, DataModule, Evaluator
from .. import GlobalValues as gl
from ..utils import jsondump
import numpy as np
class ELDMain:
	def __init__(self, mode: str, model_name: str):
		assert mode in ["train", "eval", "demo"]
		if mode != "train":
			args = ELDArgs(model_name)
			try:
				self.transformer.load_state_dict(torch.load(args.model_path))
			except:
				gl.logger.critical("No model exists!")
			self.modify_entity_embedding = args.modify_entity_embedding
			self.modify_entity_embedding_weight = args.modify_entity_embedding_weight
		else:
			try:
				args = ELDArgs.from_json("models/eld/%s_args.json" % model_name)
			except Exception:
				args = ELDArgs(model_name)
			self.epochs = args.epochs
			self.eval_per_epoch = args.eval_per_epoch
			self.model_path = args.model_path
		args.mode = mode
		self.data = DataModule(mode, args)
		self.evaluator = Evaluator(args, self.data)
		self.entity_index = {}
		self.i2e = {v: k for k, v in self.entity_index.items()}
		if mode == "test":
			self.entity_embedding = self.data.entity_embedding.weight
			self.entity_embedding_dim = self.entity_embedding.size()[-1]
		self.map_threshold = args.map_threshold
		transformer_map = {"separate": SeparateEncoderBasedTransformer, "joint": JointTransformer}
		self.transformer = transformer_map[args.transformer_mode](args.use_character_embedding, args.use_word_context_embedding, args.use_entity_context_embedding, args.use_relation_embedding, args.use_type_embedding,
		                                                          args.character_encoder, args.word_encoder, args.entity_encoder, args.relation_encoder, args.type_encoder,
		                                                          args.c_emb_dim, args.w_emb_dim, args.e_emb_dim, args.r_emb_dim, args.t_emb_dim,
		                                                          args.c_enc_dim, args.w_enc_dim, args.e_enc_dim, args.r_enc_dim, args.t_enc_dim)

		jsondump(args.to_json(), "models/eld/%s_args.json")
		gl.logger.info("ELD Model load complete")

	def train(self):
		train_batch = DataLoader(dataset=self.data.train_dataset, batch_size=32, shuffle=True, num_workers=4)
		dev_batch = DataLoader(dataset=self.data.dev_dataset, batch_size=32, shuffle=False, num_workers=4)
		tqdmloop = tqdm(range(1, self.epochs + 1))
		optimizer = torch.optim.Adam(self.transformer.parameters(), lr=1e-4, weight_decay=1e-4)
		pred_corpus = self.data.train_corpus
		gold_corpus = self.data.dev_corpus
		max_score = 0
		max_score_epoch = 0
		gl.logger.info("Train start")
		for epoch in tqdmloop:
			self.transformer.train()
			for batch in train_batch:
				optimizer.zero_grad()
				ce, cl, lwe, lwl, rwe, rwl, lee, lel, ree, rel, re, rl, te, tl = [x.to("cuda") if x is not None else None for x in batch[:-1]]  # label 빼고
				label = batch[-1]
				pred = self.transformer(ce, cl, lwe, lwl, rwe, rwl, lee, lel, ree, rel, re, rl, te, tl)
				loss = self.transformer.loss(pred, label)
				loss.backward()
				optimizer.step()
				tqdmloop.set_description("Epoch %d, Loss %.4f" % (epoch, loss))
			if epoch % self.eval_per_epoch == 0:
				self.transformer.eval()
				for batch in dev_batch:
					ce, we, ee, re, te = [x.to("cuda") if x is not None else None for x in batch[:-1]]  # label 빼고
					pred = self.transformer(ce, we, ee, re, te)
					label = batch[-1]
					pred_corpus = self.data.postprocess(pred_corpus, pred, make_copy=True)
					score = self.evaluator.evaluate(gold_corpus, pred_corpus)
					gl.logger.info("Epoch %d - Score %.4f" % (epoch, score))
					if score > max_score:
						max_score = score
						max_score_epoch = epoch
						torch.save(self.transformer.state_dict(), self.model_path)
					gl.logger.info("Best epoch %d - Score %.4f" % (max_score_epoch, max_score))

	def predict(self, data, register=True):
		pred_embedding = self.transformer(data)
		pred_embedding.repeat(len(self.entity_index))
		cos_sim = F.cosine_similarity(pred_embedding, self.entity_embedding)
		if cos_sim.max() > self.map_threshold:
			target_ind = cos_sim.argmax(dim=-1).cpu().data
			target = self.i2e[target_ind]
			if self.modify_entity_embedding:
				pred_embedding *= self.modify_entity_embedding_weight
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
