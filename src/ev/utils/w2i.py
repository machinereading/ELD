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
