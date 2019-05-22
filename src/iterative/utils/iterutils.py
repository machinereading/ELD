import torch

from ...utils import TensorUtil
from ...utils.Embedding import Embedding
class EmbeddingCalculator:
	def __init__(self, args):
		self.word_embedding = Embedding.load_embedding(args.word_embedding_type, args.word_embedding_path)
		self.entity_embedding = Embedding.load_embedding(args.entity_embedding_type, args.entity_embedding_path)
		self.word_embedding.eval()
		self.entity_embedding.eval()

	def calculate_cluster_embedding(self, clusters):
		if len(clusters) == 0: return
		self.word_embedding.cpu()
		self.entity_embedding.cpu()
		lctxe = torch.stack(tuple([c.vocab_tensors[3] for c in clusters]), dim=0)
		rctxe = torch.stack(tuple([c.vocab_tensors[4] for c in clusters]), dim=0)
		assert lctxe.size()[0] == len(clusters)
		lctx_emb = self.entity_embedding(lctxe)
		rctx_emb = self.entity_embedding(rctxe)

		cvec = (lctx_emb + rctx_emb) / 2  # -1, 100, 5, 300 -> -1, 300

		size = cvec.size()
		cvec = cvec.view([-1, size[1] * size[2], self.entity_embedding.embedding_dim])  # -1, 500, 300
		cvec = TensorUtil.nonzero_avg_stack(cvec)
		return cvec.numpy()
