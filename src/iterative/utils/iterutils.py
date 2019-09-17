import torch

from ...utils import TensorUtil, split_to_batch
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
		embeddings = []
		for batch in split_to_batch(clusters, batch_size=200):
			lctxe = torch.stack(tuple([c.vocab_tensors[3] for c in batch]), dim=0)
			rctxe = torch.stack(tuple([c.vocab_tensors[4] for c in batch]), dim=0)
			assert lctxe.size()[0] == len(batch)
			lctx_emb = self.entity_embedding(lctxe)
			rctx_emb = self.entity_embedding(rctxe)

			cvec = (lctx_emb + rctx_emb) / 2  # -1, 100, 5, 300 -> -1, 300

			size = cvec.size()
			cvec = cvec.view([-1, size[1] * size[2], self.entity_embedding.embedding_dim])  # -1, 500, 300
			cvec = TensorUtil.nonzero_avg_stack(cvec)
			for line in cvec:
				embeddings.append(line)

		return torch.stack(embeddings, dim=0).numpy()
