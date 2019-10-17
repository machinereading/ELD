import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .modules import SeparateEncoderBasedTransformer, JointTransformer, VectorTransformer
from .utils import ELDArgs, DataModule, Evaluator
from .. import GlobalValues as gl
from ..utils import jsondump

# noinspection PyMethodMayBeStatic
class ELDMain:
	def __init__(self, mode: str, model_name: str):
		self.model_name = model_name
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
		self.device = args.device = "cuda" if torch.cuda.is_available() and mode != "demo" else "cpu"

		args.mode = mode
		self.data = DataModule(mode, args)
		self.evaluator = Evaluator(args, self.data)
		self.entity_index = {}
		self.i2e = {v: k for k, v in self.entity_index.items()}
		self.use_explicit_kb_classifier = args.use_explicit_kb_classifier
		self.train_embedding = args.train_embedding
		if mode == "test":
			self.entity_embedding = self.data.entity_embedding.weight
			self.entity_embedding_dim = self.entity_embedding.size()[-1]
		self.map_threshold = args.out_kb_threshold
		transformer_map = {"separate": SeparateEncoderBasedTransformer, "joint": JointTransformer}
		self.transformer = transformer_map[args.transformer_mode] \
			(args.use_character_embedding, args.use_word_embedding, args.use_word_context_embedding, args.use_entity_context_embedding, args.use_relation_embedding, args.use_type_embedding,
			 args.character_encoder, args.word_encoder, args.word_context_encoder, args.entity_context_encoder, args.relation_encoder, args.type_encoder,
			 args.c_emb_dim, args.w_emb_dim, args.e_emb_dim, args.r_emb_dim, args.t_emb_dim,
			 args.c_enc_dim, args.w_enc_dim, args.wc_enc_dim, args.ec_enc_dim, args.r_enc_dim, args.t_enc_dim,
			 args.jamo_limit, args.word_limit, args.relation_limit).to(self.device)
		self.vector_transformer = VectorTransformer(self.transformer.max_input_dim, args.e_emb_dim, args.flags).to(self.device)
		jsondump(args.to_json(), "models/eld/%s_args.json" % model_name)
		gl.logger.info("ELD Model load complete")

	def train(self):
		batch_size = 256
		train_batch = DataLoader(dataset=self.data.train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
		dev_batch = DataLoader(dataset=self.data.dev_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
		dev_corpus = self.data.dev_corpus
		tqdmloop = tqdm(range(1, self.epochs + 1))
		discovery_optimizer = torch.optim.Adam(self.transformer.parameters(), lr=1e-3, weight_decay=1e-4)
		tensor_optimizer = torch.optim.Adam(self.vector_transformer.parameters(), lr=1e-3, weight_decay=1e-4)
		max_score = 0
		max_score_epoch = 0
		gl.logger.info("Train start")
		for epoch in tqdmloop:
			self.transformer.train()
			ne = []
			ei = []
			pr = []
			for batch in train_batch:
				discovery_optimizer.zero_grad()
				tensor_optimizer.zero_grad()
				ce, cl, we, wl, lwe, lwl, rwe, rwl, lee, lel, ree, rel, re, rl, te, tl = [x.to(self.device, torch.float) if x is not None else None for x in batch[:-3]]
				new_entity_label, ee_label, gold_entity_idx = [x.to(self.device) for x in batch[-3:]]
				kb_score, pred, attn_mask = self.transformer(ce, cl, we, wl, lwe, lwl, rwe, rwl, lee, lel, ree, rel, re, rl, te, tl)
				# pred = torch.tensor(pred, requires_grad=False).detach()
				discovery_loss_val = self.discovery_loss(kb_score, new_entity_label)
				discovery_loss_val.backward()
				discovery_optimizer.step()
				pred = self.vector_transformer(pred.detach())
				tensor_loss_val = self.out_kb_loss(pred, new_entity_label, ee_label)
				tensor_loss_val.backward()
				tensor_optimizer.step()
				tqdmloop.set_description("Epoch %d: Loss %.4f" % (epoch, discovery_loss_val + tensor_loss_val))
				ne.append(new_entity_label)
				ei.append(gold_entity_idx)
				pr.append(pred)
			self.data.update_new_entity_embedding(torch.cat(ne), torch.cat(ei), torch.cat(pr))

			if epoch % self.eval_per_epoch == 0:
				self.transformer.eval()
				self.data.reset_new_entity()
				new_ent_preds = []
				pred_entity_idxs = []
				new_entity_labels = []
				gold_entity_idxs = []
				dev_batch_start_idx = 0
				for batch in dev_batch:
					ce, cl, we, wl, lwe, lwl, rwe, rwl, lee, lel, ree, rel, re, rl, te, tl = [x.to(self.device, torch.float32) if x is not None else None for x in batch[:-3]]
					kb_score, pred, attn_mask = self.transformer(ce, cl, we, wl, lwe, lwl, rwe, rwl, lee, lel, ree, rel, re, rl, te, tl)
					kb_score = torch.sigmoid(kb_score)
					pred = self.vector_transformer(pred)
					new_entity_label, _, gold_entity_idx = [x.to(self.device) for x in batch[-3:]]
					new_ent_pred, entity_idx = self.data.predict_entity(kb_score, pred, dev_corpus.eld_get_item(slice(dev_batch_start_idx,dev_batch_start_idx+batch_size)))
					new_ent_preds += new_ent_pred
					pred_entity_idxs += entity_idx
					new_entity_labels.append(new_entity_label)
					gold_entity_idxs.append(gold_entity_idx)
				kb_expectation_score, total_score, in_kb_score, out_kb_score, no_surface_score, cluster_score, mapping_result = self.evaluator.evaluate(dev_corpus, torch.tensor(new_ent_preds).view(-1), torch.tensor(pred_entity_idxs),
				                                                                                                                                        torch.cat(new_entity_labels).cpu(), torch.cat(gold_entity_idxs).cpu())
				for score_info, (p, r, f) in [["KB expectation", kb_expectation_score],
				                              ["Total", total_score],
				                              ["in-KB", in_kb_score],
				                              ["out KB", out_kb_score],
				                              ["No surface", no_surface_score]]:
					gl.logger.info("%s score: P %.2f, R %.2f, F1 %.2f" % (score_info, p * 100, r * 100, f * 100))
				gl.logger.info("Clustering score: %.2f" % (cluster_score * 100))
				if kb_expectation_score[-1] > max_score:
					max_score = kb_expectation_score[-1]
					max_score_epoch = epoch
					torch.save(self.transformer.state_dict(), self.model_path)
				gl.logger.info("Best epoch %d - Score %.4f" % (max_score_epoch, max_score))
				jsondump(self.data.analyze(dev_corpus, torch.tensor(new_ent_preds).view(-1), torch.tensor(pred_entity_idxs), torch.cat(new_entity_labels).cpu(), torch.cat(gold_entity_idxs).cpu(),
				                           (kb_expectation_score, total_score, in_kb_score, out_kb_score, no_surface_score, cluster_score, mapping_result)), "runs/eld/%s_%d.json" % (self.model_name, epoch))

	def predict(self, data, register=True):
		self.transformer.eval()
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

	def discovery_loss(self, kb_score, new_entity_flag):
		loss = None
		closs = F.binary_cross_entropy_with_logits(kb_score.squeeze(), new_entity_flag.float())
		if loss is not None:
			loss += closs
		else:
			loss = closs
		return loss

	def out_kb_loss(self, pred, new_entity_flag, label):
		loss = sum([F.mse_loss(p, g) if t == 0 else torch.zeros_like(F.mse_loss(p, g)) for t, p, g in zip(new_entity_flag, pred, label)])
		loss += sum([F.mse_loss(p, g) if t == 1 and torch.sum(g) != 0 else torch.zeros_like(F.mse_loss(p, g)) for t, p, g in zip(new_entity_flag, pred, label)])  # zero tensor는 일단 거름
		return loss
