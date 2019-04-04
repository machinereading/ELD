import torch
import torch.nn as nn
import numpy as np
import sys
if sys.version_info >= (3, 6):
	from allennlp.modules.elmo import Elmo, batch_to_ids

from pytorch_pretrained_bert.modeling import *
from ...utils import readfile
class Embedding(nn.Module):
	def __init__(self, embedding_type, embedding_path, device):
		super(Embedding, self).__init__()
		self.embedding_type = None
		self.w2i = None
		self.embedding = None
		self.load_embedding(embedding_type, embedding_path)
		self.embedding.to(device)
	
	def forward(self, word_batch):
		return self.embedding(word_batch)

	def load_embedding(self, embedding_type, embedding_path):
		embedding_type = embedding_type.lower()
		emb_list = ["glove", "bert"]
		if sys.version_info >= (3, 6):
			emb_list.append("elmo")

		if embedding_type not in emb_list:
			raise ValueError("Embedding type must be one of %s" % ", ".join(emb_list))

		self.embedding_type = embedding_type
		if embedding_type == "glove":
			w2i = {w: i+2 for i, w in enumerate(readfile(embedding_path+".word"))}
			e = np.load(embedding_path+".npy")
			e = np.vstack([np.zeros([2, e.shape[1]]), e])
			glove_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(e))
			self.embedding = glove_embedding
		elif embedding_type == "elmo":
			pass
		elif embedding_type == "bert":
			self.embedding = Bert.from_pretrained('bert-base-multilingual-cased')

class Bert(BertPreTrainedModel):
	def __init__(self, config):
		super(BertPretrained, self).__init__(config)
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