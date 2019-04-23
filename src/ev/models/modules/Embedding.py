import torch
import torch.nn as nn
import numpy as np
import sys
if sys.version_info >= (3, 6):
	from allennlp.modules.elmo import Elmo, batch_to_ids

from pytorch_pretrained_bert.modeling import *
from ....utils import readfile
class Embedding(nn.Module):
	def __init__(self, embedding_type, embedding_path):
		super(Embedding, self).__init__()
		self.embedding_type = None
		self.w2i = None
		self.embedding = None
		self.load_embedding(embedding_type, embedding_path)

	
	def forward(self, word_batch, **kwargs):
		
		emb = self.embedding(word_batch, **kwargs)
		# return emb
		if self.embedding_type == "bert":
			# print(type(emb), emb.size())
			# print(type(emb[0]), len(emb), emb[-1].size())
			return emb[0][-1]
		else:
			return emb

	def load_embedding(self, embedding_type, embedding_path):
		embedding_type = embedding_type.lower()
		emb_list = ["glove", "bert"]
		if sys.version_info >= (3, 6):
			emb_list.append("elmo")

		if embedding_type not in emb_list:
			raise ValueError("Embedding type must be one of %s" % ", ".join(emb_list))

		self.embedding_type = embedding_type
		if embedding_type == "glove":
			e = np.load(embedding_path+".npy")
			e = np.vstack([e, np.zeros([2, e.shape[1]])])
			glove_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(e))
			self.embedding = glove_embedding
			self.embedding_dim = e.shape[1]
		elif embedding_type == "elmo":
			pass
		elif embedding_type == "bert":
			self.embedding = BertModel.from_pretrained('bert-base-multilingual-cased')
			self.embedding_dim = 768


class Bert(BertPreTrainedModel):
	def __init__(self, config):
		super(Bert, self).__init__(config)
		self.bert = BertModel(config)

	def forward(self, input):
		return self.bert(input)

def bert_tokenizer(text):
	orig_tokens = text
	bert_tokens = []
	orig_to_tok_map = []
	bert_tokens.append("[CLS]")
	for orig_token in orig_tokens:
		orig_to_tok_map.append(len(bert_tokens))
		bert_tokens.extend(tokenizer.tokenize(orig_token))
	bert_tokens.append("[SEP]")
	
	return orig_tokens, bert_tokens, orig_to_tok_map