import torch
from ...utils import TensorUtil
class EmbeddingCalculator():
	def __init__(self, word_embedding, entity_embedding):
		self.word_embedding = word_embedding
		self.entity_embedding = entity_embedding

	def calculate_cluster_embedding(self, cluster):
		tok_emb = torch.zeros([self.entity_embedding.embedding_dim])
		tok_count = 0
		_, _, _, lctxe, rctxe, l, _ = cluster.vocab_tensors
		lctx_emb = self.entity_embedding(lctxe)
		rctx_emb = self.entity_embedding(rctxe)
		avg = (TensorUtil.nonzero_avg_stack(lctxe) + TensorUtil.nonzero_avg_stack(rctxe)) / 2
		return avg

			