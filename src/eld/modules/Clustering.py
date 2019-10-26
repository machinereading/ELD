import torch
import torch.nn as nn
import torch.nn.functional as F

# noinspection PyMethodMayBeStatic
@DeprecationWarning
class SynsetMine(nn.Module):
	"""
	Reference paper: Jiaming Shen et al., Mining Entity Synonyms with Efficient Neural Set Generation, AAAI, 2019
	has same architecture except embedding layer of token embedding. it is replaced with entity embedding, which is given as input
	NOT USED
	"""
	def __init__(self, embedding_input_dim):
		super(SynsetMine, self).__init__()
		self.set_scorer_embedding_transformer = nn.Sequential(nn.Linear(embedding_input_dim, 250), nn.ReLU(), nn.Dropout(), nn.Linear(250, 250), nn.ReLU(), nn.Dropout(p=0.5))
		self.set_scorer_post_transformer = nn.Sequential(nn.Linear(250, 500), nn.ReLU(), nn.Dropout(), nn.Linear(500, 250), nn.ReLU(), nn.Dropout(), nn.Linear(250, 1))

	def forward(self, set_tensors, tensors):
		"""
		set_tensors: batch_size * set_size * embedding_size
		tensors: batch_size * embedding_size
		"""
		original_set_transform = self.set_scorer_embedding_transformer(set_tensors)
		original_set_transform_sum = torch.sum(original_set_transform, 1)
		original_set_score = self.set_scorer_post_transformer(original_set_transform_sum)

		additional_set_tensors = [torch.cat([x, y], dim=0) for x, y in zip(set_tensors, tensors)]
		additional_set_transform = self.set_scorer_embedding_transformer(additional_set_tensors)
		additional_set_transform_sum = torch.sum(additional_set_transform, 1)
		additional_set_score = self.set_scorer_post_transformer(additional_set_transform_sum)

		return additional_set_score - original_set_score

	def loss(self, pred, label):
		return F.binary_cross_entropy_with_logits(pred, label)
