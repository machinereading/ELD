import torch
import torch.nn as nn
from allennlp.modules.elmo import Elmo, batch_to_ids
from pytorch_pretrained_bert import BertTokenizer
class Embedding(nn.Module):
	def __init__(self, args):
		super(Embedding, self).__init__()


	def initialize_embedding_model(self, embedding_type):
		pass