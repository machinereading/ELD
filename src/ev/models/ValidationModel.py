import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .modules import ContextEmbedder, Transformer, Scorer
from ...utils import KoreanUtil
from ...utils.Embedding import Embedding

class ValidationModel(nn.Module):
	def __init__(self, args):
		super(ValidationModel, self).__init__()
		self.pretrain_epoch = args.pretrain_epoch
		self.pretrain_transformer = True
		self.kb = None
		self.window_size = args.ctx_window_size
		self.target_device = args.device
		self.chunk_size = args.chunk_size
		print(self.target_device)
		self.we = Embedding.load_embedding(args.word_embedding_type, args.word_embedding_path).to(self.target_device)
		self.ee = Embedding.load_embedding(args.entity_embedding_type, args.entity_embedding_path).to(self.target_device)
		self.ce = nn.Embedding(KoreanUtil.jamo_len + len(KoreanUtil.alpha) + 1, args.char_embedding_dim).to(self.target_device)
		self.we_dim = self.we.embedding_dim
		self.ee_dim = self.ee.embedding_dim
		self.ce_dim = args.char_embedding_dim

		self.transformer_type = args.transformer

		self.use_explicit_scorer = getattr(args, "use_explicit_scorer", False)
		# sequential encoding
		self.jamo_embedder = ContextEmbedder.CNNEmbedder(args.max_jamo, 2, 2)
		self.cw_embedder = ContextEmbedder.BiContextEmbedder(args.er_model, self.we_dim, args.er_output_dim)
		self.ce_embedder = ContextEmbedder.BiContextEmbedder(args.el_model, self.ee_dim, args.el_output_dim)
		args.jamo_embed_dim = (args.char_embedding_dim - 2) // 2 * 2

		if self.use_explicit_scorer:
			self.token_scorer = Scorer.EmbedScorer(args, args.jamo_embed_dim, args.er_output_dim, args.el_output_dim)

		self.cluster_transformer = Transformer.get_transformer(args)
		# final prediction layer
		self.predict = nn.Sequential(
				nn.Linear(args.transform_dim, 100),
				nn.ReLU(),
				nn.Linear(100, 1),
				nn.Sigmoid()
		)

	def forward(self, jamo_index, cluster_word_lctx, cluster_word_rctx, cluster_entity_lctx, cluster_entity_rctx, size):
		"""
		Input
			jamo_index: Tensor of batch_size * max_vocab_size * max jamo size
			cluster_word/entity_lctx/rctx: Tensor of batch size * max vocab size * window size
		Output
			Confidence score?
		"""
		# cluster -> representation
		# print(jamo_index.size(), cluster_word_lctx.size(), cluster_entity_lctx.size())
		# split into 100
		chunks = jamo_index.size()[0] * jamo_index.size()[1] // self.chunk_size
		size = (size - 1) // self.chunk_size + 1
		jamo_size = jamo_index.size()[-1]

		# print(jamo_index.size())
		# for item in torch.chunk(jamo_index.view(-1, self.chunk_size, jamo_size), chunks, dim=1):
		# 	print(item.size(), torch.sum(item))
		# print(torch.stack([x for x in torch.chunk(jamo_index.view(-1, self.chunk_size, jamo_size), chunks, dim=1) if torch.sum(x) != 0]).size())

		# jamo_index = torch.stack([x for x in jamo_index.view(-1, self.chunk_size, jamo_size) if torch.sum(x) != 0]).view(-1, jamo_size)
		# cluster_word_lctx = torch.stack([x for x in cluster_word_lctx.view(-1, self.chunk_size, self.window_size) if torch.sum(x) != 0]).view(-1, self.window_size)
		# cluster_word_rctx = torch.stack([x for x in cluster_word_rctx.view(-1, self.chunk_size, self.window_size) if torch.sum(x) != 0]).view(-1, self.window_size)
		# cluster_entity_lctx = torch.stack([x for x in cluster_entity_lctx.view(-1, self.chunk_size, self.window_size) if torch.sum(x) != 0]).view(-1, self.window_size)
		# cluster_entity_rctx = torch.stack([x for x in cluster_entity_rctx.view(-1, self.chunk_size, self.window_size) if torch.sum(x) != 0]).view(-1, self.window_size)
		jamo_index = self.nonzero_stack(jamo_index, jamo_size)
		cwl = self.nonzero_stack(cluster_word_lctx, self.window_size)
		cwr = self.nonzero_stack(cluster_word_rctx, self.window_size)
		cel = self.nonzero_stack(cluster_entity_lctx, self.window_size)
		cer = self.nonzero_stack(cluster_entity_rctx, self.window_size)

		# print(size)
		# print(jamo_index.size(), cluster_word_lctx.size(), cluster_word_rctx.size(), cluster_entity_lctx.size(), cluster_entity_rctx.size(), size.size())  # 100 * non-empty batch size * jamo size
		assert jamo_index.size()[0] == cwl.size()[0] == cwr.size()[0] == cel.size()[0] == cer.size()[0], "Size mismatch: %d, %d, %d, %d, %d" % (jamo_index.size()[0], cwl.size()[0], cwr.size()[0], cel.size()[0], cer.size()[0])
		# assert torch.sum(size) == jamo.index.size()[0]
		c_embedding = self.ce(jamo_index).view(-1, jamo_size, self.ce_dim)  # batch_size * max_vocab_size * max_jamo_size * jamo_embedding
		# bert attention mask?
		# batch_size * max_vocab_size * window * embedding_size
		wlctx = self.we(cwl).view(-1, self.window_size, self.we_dim)
		wrctx = self.we(cwr).view(-1, self.window_size, self.we_dim)
		elctx = self.ee(cel).view(-1, self.window_size, self.ee_dim)
		erctx = self.ee(cer).view(-1, self.window_size, self.ee_dim)
		# print(c_embedding.size(), wlctx.size(), elctx.size())

		c_embedding = self.jamo_embedder(c_embedding)
		w_embedding = self.cw_embedder(wlctx, wrctx)
		e_embedding = self.ce_embedder(elctx, erctx)

		if self.use_explicit_scorer:
			token_score = self.token_scorer(c_embedding, w_embedding, e_embedding)  # n * chunk * 1
			# print(token_score, c_embedding, token_score * c_embedding)
			c_embedding *= token_score
			w_embedding *= token_score
			e_embedding *= token_score

		c_embedding = c_embedding.view(-1, self.chunk_size, c_embedding.size()[-1])
		w_embedding = w_embedding.view(-1, self.chunk_size, w_embedding.size()[-1])  # (batch_size * max_vocab_size) * embedding size
		e_embedding = e_embedding.view(-1, self.chunk_size, e_embedding.size()[-1])

		cluster_representation = self.cluster_transformer(c_embedding, w_embedding, e_embedding)  # batch size * cluster representation size
		# print(cluster_representation)
		# prediction - let's try simple FFNN this time
		prediction = self.predict(cluster_representation)
		# print("Prediction:", prediction)
		return prediction

	def loss(self, prediction, label):
		# print(prediction)
		# print(label)
		return F.binary_cross_entropy(prediction, label)

	def pretrain(self, dataset):
		# pretrain transformer
		if self.pretrain_transformer:
			for epoch in tqdm(range(self.pretrain_epoch)):
				pass

		for param in self.cluster_transformer:
			param.requires_grad = False

	# def _split_with_pad(self, tensor, size):
	# 	tensor_size = tensor.size()
	# 	if size < self.chunk_size:
	# 		if tensor_size[0] < self.chunk_size:
	# 			yield torch.stack([tensor, torch.zeros([self.chunk_size - tensor_size[0], tensor_size[1]])])
	# 			return
	# 		yield tensor[:self.chunk_size, :]
	# 		return
	# 	slices = tensor_size[0] // self.chunk_size
	# 	for i in range(slices):
	# 		yield tensor[i * self.chunk_size:(i + 1) * self.chunk_size, :]
	# 	yield torch.stack([tensor[slices * self.chunk_size:, :],
	# 	                   torch.zeros([self.chunk_size - (size - slices * self.chunk_size), tensor_size[1]])])
	# 	return

	def nonzero_stack(self, tensor, s):
		try:
			return torch.stack(tuple([x for x in tensor.view(-1, self.chunk_size, s) if torch.sum(x) != 0])).view(-1, s)
		except:
			return torch.zeros_like(tensor)
