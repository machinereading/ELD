import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils import KoreanUtil
from . import BiContextEmbedding, CNNEmbedding


class TokenEmbeddingModule(nn.Module):
	def __init__(self, args: EmbeddingArgs):
		super(TokenEmbeddingModule, self).__init__()
		self.kb = None
		self.window_size = args.ctx_window_size
		self.target_device = args.device
		self.chunk_size = args.chunk_size
		self.we = nn.Embedding.load_embedding(args.word_embedding_type, args.word_embedding_path).to(self.target_device)
		self.ee = nn.Embedding.load_embedding(args.entity_embedding_type, args.entity_embedding_path).to(self.target_device)
		self.ce = nn.Embedding(KoreanUtil.jamo_len + len(KoreanUtil.alpha) + 1, args.char_embedding_dim).to(self.target_device)
		self.we_dim = self.we.embedding_dim
		self.ee_dim = self.ee.embedding_dim
		self.ce_dim = args.char_embedding_dim

		# sequential encoding
		self.jamo_embedder = CNNEmbedding(args.max_jamo, 2, 2)
		self.cw_embedder = BiContextEmbedding(args.er_model, self.we_dim, args.er_output_dim)
		self.ce_embedder = BiContextEmbedding(args.el_model, self.ee_dim, args.el_output_dim)
		args.jamo_embed_dim = (args.char_embedding_dim - 2) // 2 * 2

		# final prediction layer
		self.predict = nn.Sequential(
				nn.Linear(args.transform_dim, 100),
				nn.ReLU(),
				nn.Linear(100, 1),
				nn.Sigmoid()
		)
	def forward(self, j, lctx, rctx):
		j_embedding = self.jamo_embedding(j)
		lctx_w_embedding = self.word_embedding(lctx)
		rctx_w_embedding = self.word_embedding(rctx)

		j_encoded = self.jamo_encoder(j_embedding)
		context_encoded, _ = self.context_lstm(lctx_w_embedding)


	def loss(self, output_embedding, target_embedding, negative_sample_embedding):
		# mseloss?
		# embedding을 통해 negative sample set에서 예측하는 방향으로 할지, 아니면 그냥 embedding에서 멀어지는 식으로 학습할지
		return F.mse_loss(output_embedding, target_embedding) - self.negative_loss_modifier * sum([F.mse_loss(target_embedding, x) for x in negative_sample_embedding])
