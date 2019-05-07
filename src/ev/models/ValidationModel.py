import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ...utils import readfile, KoreanUtil
from ... import GlobalValues as gl
from .modules import ContextEmbedder, Transformer
from .modules.Embedding import Embedding


class ValidationModel(nn.Module):
	def __init__(self, args):
		super(ValidationModel, self).__init__()
		self.pretrain_epoch = args.pretrain_epoch
		self.pretrain_transformer = True
		self.kb = None
		self.window_size = args.ctx_window_size
		self.target_device = args.device
		self.chunk_size = args.chunk_size

		self.we = Embedding(args.word_embedding_type, args.word_embedding_path).to(self.target_device)
		self.ee = Embedding(args.entity_embedding_type, args.entity_embedding_path).to(self.target_device)
		self.ce = nn.Embedding(KoreanUtil.jamo_len + len(KoreanUtil.alpha) + 1, args.char_embedding_dim).to(self.target_device)
		self.we_dim = self.we.embedding_dim
		self.ee_dim = self.ee.embedding_dim
		self.ce_dim = args.char_embedding_dim
		
		self.transformer_type = args.transformer
		
		self.encode_sequence = args.encode_sequence

		# sequential encoding
		if self.encode_sequence:
			self.jamo_embedder = ContextEmbedder.CNNEmbedder(args.max_jamo, 2, 2)
			self.cw_embedder = ContextEmbedder.BiContextEmbedder(args.er_model, self.we_dim, args.er_output_dim)
			self.ce_embedder = ContextEmbedder.BiContextEmbedder(args.el_model, self.ee_dim, args.el_output_dim)
			args.jamo_embed_dim = (args.char_embedding_dim - 2) // 2 * 2
		else:
			args.jamo_embed_dim = args.char_embedding_dim

		self.cluster_transformer = Transformer.get_transformer(args)
		# final prediction layer
		self.predict = nn.Sequential(
			nn.Linear(args.transform_dim, 100),
			nn.ReLU(),
			nn.Linear(100, 1),
			nn.Sigmoid()
		)
		try:
			self.cluster_transformer.load_state_dict(torch.load(args.transformer_model_path))
			self.pretrain_transformer = False
		except:
			gl.logger.info("Failed to load transformer from %s" % args.transformer_model_path)

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
		size = (size-1) // self.chunk_size + 1
		jamo_size = jamo_index.size()[-1]

		# print(jamo_index.size())
		# for item in torch.chunk(jamo_index.view(-1, self.chunk_size, jamo_size), chunks, dim=1):
		# 	print(item.size(), torch.sum(item))
		# print(torch.stack([x for x in torch.chunk(jamo_index.view(-1, self.chunk_size, jamo_size), chunks, dim=1) if torch.sum(x) != 0]).size())
		nonzero_stack = lambda tensor, size: torch.stack([x for x in tensor.view(-1, self.chunk_size, size) if torch.sum(x) != 0]).view(-1, size)

		# jamo_index = torch.stack([x for x in jamo_index.view(-1, self.chunk_size, jamo_size) if torch.sum(x) != 0]).view(-1, jamo_size)
		# cluster_word_lctx = torch.stack([x for x in cluster_word_lctx.view(-1, self.chunk_size, self.window_size) if torch.sum(x) != 0]).view(-1, self.window_size)
		# cluster_word_rctx = torch.stack([x for x in cluster_word_rctx.view(-1, self.chunk_size, self.window_size) if torch.sum(x) != 0]).view(-1, self.window_size)
		# cluster_entity_lctx = torch.stack([x for x in cluster_entity_lctx.view(-1, self.chunk_size, self.window_size) if torch.sum(x) != 0]).view(-1, self.window_size)
		# cluster_entity_rctx = torch.stack([x for x in cluster_entity_rctx.view(-1, self.chunk_size, self.window_size) if torch.sum(x) != 0]).view(-1, self.window_size)
		jamo_index = nonzero_stack(jamo_index, jamo_size)
		cluster_word_lctx = nonzero_stack(cluster_word_lctx, self.window_size)
		cluster_word_rctx = nonzero_stack(cluster_word_rctx, self.window_size)
		cluster_entity_lctx = nonzero_stack(cluster_entity_lctx, self.window_size)
		cluster_entity_rctx = nonzero_stack(cluster_entity_rctx, self.window_size)

		# print(size)
		# print(jamo_index.size(), cluster_word_lctx.size(), cluster_word_rctx.size(), cluster_entity_lctx.size(), cluster_entity_rctx.size(), size.size()) # 100 * non-empty batch size * jamo size
		assert jamo_index.size()[0] == cluster_word_lctx.size()[0] == cluster_word_rctx.size()[0] == cluster_entity_lctx.size()[0] == cluster_entity_rctx.size()[0], "Size mismatch"
		# assert torch.sum(size) == jamo.index.size()[0]
		c_embedding = self.ce(jamo_index).view(-1, jamo_size, self.ce_dim) # batch_size * max_vocab_size * max_jamo_size * jamo_embedding
		voca_size = jamo_index.size()[1]
		# bert attention mask?
		wlctx = self.we(cluster_word_lctx.view(-1, self.window_size)).view(-1, self.window_size, self.we_dim) # batch_size * max_vocab_size * window * embedding_size
		wrctx = self.we(cluster_word_rctx.view(-1, self.window_size)).view(-1, self.window_size, self.we_dim)
		elctx = self.ee(cluster_entity_lctx).view(-1, self.window_size, self.ee_dim)
		erctx = self.ee(cluster_entity_rctx).view(-1, self.window_size, self.ee_dim)
		# print(c_embedding.size(), wlctx.size(), elctx.size())
		
		if self.encode_sequence:
			c_embedding = self.jamo_embedder(c_embedding)
			c_embedding = c_embedding.view(-1, self.chunk_size, c_embedding.size()[-1])
			w_embedding = self.cw_embedder(wlctx, wrctx) # (batch_size * max_vocab_size) * embedding size
			# print(w_embedding.size())
			w_embedding = w_embedding.view(-1, self.chunk_size, w_embedding.size()[-1]) # batch_size * max_vocab_size * embedding size - 각각의 token마다 embedding dimension의 context embedding 하나씩 들고있음
			# print(w_embedding.size())
			e_embedding = self.ce_embedder(elctx, erctx)
			e_embedding = e_embedding.view(-1, self.chunk_size, e_embedding.size()[-1])
		else:
			w_embedding = torch.cat([wlctx, wrctx], 2)
			e_embedding = torch.cat([elctx, erctx], 2)

		
		cluster_representation = self.cluster_transformer(c_embedding, w_embedding, e_embedding) # batch size * cluster representation size
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

	def _split_with_pad(self, tensor, size):
		tensor_size = tensor.size()
		if size < self.chunk_size:
			if tensor_size[0] < self.chunk_size:
				yield torch.stack([tensor, torch.zeros([self.chunk_size - tensor_size[0], tensor_size[1]])])
				return
			yield tensor[:self.chunk_size, :]
			return
		slices = tensor_size[0] // self.chunk_size
		for i in range(slices):
			yield tensor[i*self.chunk_size:(i+1)*self.chunk_size, :]
		yield torch.stack([tensor[slices * self.chunk_size:, :], torch.zeros([self.chunk_size - (size - slices * self.chunk_size), tensor_size[1]])])
		return