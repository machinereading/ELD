import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from . import zoo
import math
from sklearn.metrics import confusion_matrix

def initialize_weights(moduleList, itype="xavier"):
	""" Initialize a list of modules

	:param moduleList: a list of nn.modules
	:type moduleList: list
	:param itype: name of initialization method
	:type itype: str
	:return: None
	:rtype: None
	"""
	assert itype == 'xavier', 'Only Xavier initialization supported'

	for moduleId, module in enumerate(moduleList):
		if hasattr(module, '_modules') and len(module._modules) > 0:
			# Iterate again
			initialize_weights(module, itype)
		else:
			# Initialize weights
			name = type(module).__name__
			# If linear or embedding
			if name == 'Embedding' or name == 'Linear':
				fanIn = module.weight.data.size(0)
				fanOut = module.weight.data.size(1)

				factor = math.sqrt(2.0/(fanIn + fanOut))
				weight = torch.randn(fanIn, fanOut) * factor
				module.weight.data.copy_(weight)

			# Check for bias and reset
			if hasattr(module, 'bias') and hasattr(module.bias, 'data'):
				module.bias.data.fill_(0.0)

class SSPM2(nn.Module):
	def __init__(self, params):
		super(SSPM, self).__init__()
		self.initialize(params)

		if params['loss_fn'] == "cross_entropy":
			self.criterion = nn.NLLLoss()
		elif params['loss_fn'] == "max_margin":
			self.criterion = nn.MultiMarginLoss(margin=params['margin'])
		elif params['loss_fn'] in ["margin_rank", "self_margin_rank"]:
			self.criterion = nn.MarginRankingLoss(margin=params['margin'])
		elif params['loss_fn'] == "self_margin_rank_bce":
			self.criterion = nn.BCEWithLogitsLoss()

		# TODO: avoid the following self.params = params
		self.params = params
		# transfer parameters to self, therefore we have self.modelName
		for key, val in self.params.items():
			setattr(self, key, val)

		self.temperature = params["T"]  # use for temperature scaling

	def initialize(self, params):
		""" Initialize model components

		:param params: a dictionary containing all model specifications
		:type params: dict
		:return: None
		:rtype: None
		"""
		modelParts = zoo.select_model(params)
		flags = ['node_embedder', 'node_postEmbedder', 'node_pooler', 'edge_embedder', 'edge_postEmbedder',
				 'edge_pooler', 'combiner', 'scorer']

		# refine flags
		for flag in flags:
			if flag not in modelParts:
				print('Missing: %s' % flag)
			else:
				setattr(self, flag, modelParts[flag])

		setattr(self, "context_embedder", zoo.select_context_embedder(params))
		# define node transform as composition
		self.nodeTransform = lambda x: self.node_postEmbedder(self.node_embedder(x))
		self.edgeTransform = lambda x: self.edge_postEmbedder(self.edge_embedder(x))

		# Initialize the parameters with xavier method
		modules = ['node_embedder', 'node_postEmbedder', 'edge_embedder', 'edge_postEmbedder', 'combiner', 'scorer']
		modules = [getattr(self, mod) for mod in modules if hasattr(self, mod)]
		initialize_weights(modules, 'xavier')

		if params['pretrained_embedding'] != "none":
			pretrained_embedding = params['embedding'].vectors
			padding_embedding = np.zeros([1, pretrained_embedding.shape[1]])
			pretrained_embedding = np.row_stack([padding_embedding, pretrained_embedding])
			self.node_embedder.weight.data.copy_(torch.from_numpy(pretrained_embedding))
			if params['embed_fine_tune'] == 0:  # fix embedding without fine-tune
				self.node_embedder.weight.requires_grad = False