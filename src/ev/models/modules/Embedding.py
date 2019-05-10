import sys

import numpy as np

if sys.version_info >= (3, 6):
	pass

from pytorch_pretrained_bert.modeling import *

class Embedding(nn.Module):
	embedding_path_dict = {}
	def __init__(self, embedding_type):
		super(Embedding, self).__init__()
		self.embedding_type = embedding_type
		self.w2i = None
		self.embedding = None

	def forward(self, word_batch, **kwargs):

		emb = self.embedding(word_batch, **kwargs)
		# return emb
		if self.embedding_type == "bert":
			# print(type(emb), emb.size())
			# print(type(emb[0]), len(emb), emb[-1].size())
			return emb[0][-1]
		else:
			return emb
	@classmethod
	def load_embedding(cls, embedding_type, embedding_path):
		if embedding_path in cls.embedding_path_dict:
			return Embedding.embedding_path_dict[embedding_path]

		embedding_type = embedding_type.lower()
		emb_list = ["glove", "bert"]
		if sys.version_info >= (3, 6):
			emb_list.append("elmo")

		if embedding_type not in emb_list:
			raise ValueError("Embedding type must be one of %s" % ", ".join(emb_list))

		embedding = cls(embedding_type)
		if embedding_type == "glove":
			e = np.load(embedding_path + ".npy")
			e = np.vstack([e, np.zeros([2, e.shape[1]])])
			glove_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(e))
			embedding.embedding = glove_embedding
			embedding.embedding_dim = e.shape[1]
		elif embedding_type == "elmo":
			pass
		elif embedding_type == "bert":
			embedding.embedding = BertModel.from_pretrained('bert-base-multilingual-cased')
			embedding.embedding_dim = 768
		cls.embedding_path_dict[embedding_path] = embedding
		return embedding

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
