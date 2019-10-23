import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

from transformers.modeling_bert import BertConfig, BertEncoder

module = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}

class BiContextEncoder(nn.Module):
	def __init__(self, model, input_dim, output_dim, use_attention=True):
		super(BiContextEncoder, self).__init__()
		# self.lctx_model = ModelFactory.load_model(args.er_model, input_dim, args.er_output_dim)
		self.hidden_size = output_dim
		self.lctx_model = module[model.lower()](input_size=input_dim, hidden_size=self.hidden_size, batch_first=True, bidirectional=False)
		self.rctx_model = module[model.lower()](input_size=input_dim, hidden_size=self.hidden_size, batch_first=True, bidirectional=False)
		self.use_attention = use_attention

	def forward(self, lctx, lctxl, rctx, rctxl, *args):
		# print(lctx.size(), lctxl)
		lctx = rnn.pack_padded_sequence(lctx, lctxl, batch_first=True, enforce_sorted=False)
		rctx = rnn.pack_padded_sequence(rctx, rctxl, batch_first=True, enforce_sorted=False)

		lctx_emb, (lctx_hidden, _) = self.lctx_model(lctx)
		lctx_emb, _ = rnn.pad_packed_sequence(lctx_emb, batch_first=True)
		rctx_emb, (rctx_hidden, _) = self.rctx_model(rctx)
		rctx_emb, _ = rnn.pad_packed_sequence(rctx_emb, batch_first=True)
		# print(lctx_emb.size(), lctx_hidden.size())
		if self.use_attention:
			lctx_emb, _ = self.attention(lctx_emb, lctx_hidden)
			rctx_emb, _ = self.attention(rctx_emb, rctx_hidden)

		return F.relu(torch.cat([lctx_emb, rctx_emb], dim=-1))

	def attention(self, lstm_output, final_state):
		# code from https://medium.com/platfarm/%EC%96%B4%ED%85%90%EC%85%98-%EB%A9%94%EC%BB%A4%EB%8B%88%EC%A6%98%EA%B3%BC-transfomer-self-attention-842498fd3225
		hidden = final_state.view(-1, self.hidden_size, 1)
		attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
		soft_attn_weights = F.softmax(attn_weights, 1)
		context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
		return context, soft_attn_weights

class CNNEncoder(nn.Module):
	def __init__(self, in_dim, in_channel, out_channel, kernel_size):
		super(CNNEncoder, self).__init__()
		self.enc = nn.Sequential(
				nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size),
				nn.MaxPool1d(2)
		)
		self.out_size = ((out_channel * (in_dim - (kernel_size - 1))) - 2) // 2 + 1

	def forward(self, input_tensor, *args):  # batch * in_channel * L -> batch * out_channel * Lout
		return self.enc(input_tensor).view(input_tensor.size(0), -1)

class RNNEncoder(nn.Module):
	def __init__(self, input_dim, output_dim, use_attention=True):
		super(RNNEncoder, self).__init__()
		self.hidden_size = output_dim
		self.encoder = nn.LSTM(input_size=input_dim, hidden_size=self.hidden_size, batch_first=True, bidirectional=True)
		self.use_attention = use_attention

	def forward(self, tensor, length, *args):
		seq = rnn.pack_padded_sequence(tensor, length, batch_first=True, enforce_sorted=False)

		enc, hidden = self.encoder(seq)
		pad_enc = rnn.pad_packed_sequence(enc[0], batch_first=True)  # TODO

		if self.use_attention:
			enc, _ = self.attention(enc, hidden)
		return F.relu(enc)

	def attention(self, lstm_output, final_state):
		# code from https://medium.com/platfarm/%EC%96%B4%ED%85%90%EC%85%98-%EB%A9%94%EC%BB%A4%EB%8B%88%EC%A6%98%EA%B3%BC-transfomer-self-attention-842498fd3225
		hidden = final_state.view(-1, self.hidden_size, 1)
		attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
		soft_attn_weights = F.softmax(attn_weights, 1)
		context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
		return context, soft_attn_weights

class SelfAttentionEncoder(nn.Module):
	# using https://github.com/huggingface/pytorch-transformers
	def __init__(self, input_size, hidden_layers, num_attention_heads, features, output_attentions=True, output_dim=None):
		super(SelfAttentionEncoder, self).__init__()
		# assert hidden_size % attention_heads == 0
		self.config = BertConfig(hidden_size=input_size, num_hidden_layers=hidden_layers, num_attention_heads=num_attention_heads, output_attentions=output_attentions)
		self.input_size = input_size
		self.hidden_layers = hidden_layers
		self.num_attention_heads = num_attention_heads
		self.separate = features
		self.encoder = BertEncoder(self.config)
		self.apply_ffnn = output_dim is not None
		if self.apply_ffnn:
			self.ffnn = nn.Linear(input_size * features, output_dim)

	def forward(self, hidden_state, attention_mask=None, head_mask=None, *args):
		if attention_mask is None:
			attention_mask = torch.ones([hidden_state.size(0), self.num_attention_heads, self.separate, self.separate]).to(hidden_state.device)
		# print(hidden_state.size())
		hidden_state = hidden_state.view(-1, self.separate, self.input_size)
		if head_mask is None:
			head_mask = [None] * self.config.num_hidden_layers
		# attention_mask = attention_mask.view(-1, self.separate, self.input_size)
		# print(hidden_state.size())
		output = self.encoder(hidden_state, attention_mask, head_mask)
		if self.apply_ffnn:
			output = F.relu(self.ffnn(F.dropout(output[0].view(-1, self.separate * self.input_size))))
		return output

class Ident(nn.Module):
	def __init__(self, *args, **kwargs):
		super(Ident, self).__init__()

	def forward(self, tensor, *args):
		return tensor

class FFNNEncoder(nn.Module):
	def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
		super(FFNNEncoder, self).__init__()
		layers = [nn.Linear(input_dim, output_dim), nn.ReLU(), nn.Dropout()] if num_layers < 2 else \
			[nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout()] + [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout()] * (num_layers - 2) + [nn.Linear(hidden_dim, output_dim), nn.ReLU(), nn.Dropout()]
		self.nn = torch.nn.Sequential(*layers)

	def forward(self, input_tensor, *args):
		return self.nn(input_tensor)
