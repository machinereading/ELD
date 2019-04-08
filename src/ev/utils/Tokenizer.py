from pytorch_pretrained_bert import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
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

def bert_tokenize(text):
	orig_tokens, bert_tokens, orig_to_tok_map = bert_tokenizer(text)
	tokenized_text = bert_tokens
	converted = tokenizer.convert_tokens_to_ids(tokenized_text)
	return converted


class Tokenizer():
	def __init__(self, mode, w2i=None):
		assert mode in ["glove", "elmo", "bert"]
		self.tokenize_mode = mode
		if w2i is not None:
			self.w2i = lambda x: w2i[x] if x in w2i else 0

	def tokenize(self, sent):
		"""
			tokenize sentence
			input:
				sent: list of vocab
			output: list of vocab
		"""
		sent = [x.surface for x in sent]
		if self.tokenize_mode == "glove":
			return [self.w2i(x) for x in sent]
		if self.tokenize_mode == "bert":
			return bert_tokenize(sent)

	def __call__(self, sent):
		return self.tokenize(sent)