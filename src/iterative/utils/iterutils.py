import torch

from ...utils import TensorUtil

class EmbeddingCalculator():
	def __init__(self, word_embedding, entity_embedding):
		if torch.cuda.is_available():
			self.device = "cuda"
		else:
			self.device = "cpu"
		self.word_embedding = word_embedding
		self.entity_embedding = entity_embedding

	def calculate_cluster_embedding(self, clusters):
		lctxe = torch.stack([c.vocab_tensors[3] for c in clusters], dim=0)
		rctxe = torch.stack([c.vocab_tensors[4] for c in clusters], dim=0)
		assert lctxe.size()[0] == len(clusters)
		lctx_emb = self.entity_embedding(lctxe)
		rctx_emb = self.entity_embedding(rctxe)

		cvec = (lctx_emb + rctx_emb) / 2  # -1, 100, 5, 300 -> -1, 300

		size = cvec.size()
		cvec = cvec.view([-1, size[1] * size[2], self.entity_embedding.embedding_dim])  # -1, 500, 300
		cvec = TensorUtil.nonzero_avg_stack(cvec)
		return cvec.cpu().numpy()
