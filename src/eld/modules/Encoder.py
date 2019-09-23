import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
from pytorch_transformers.modeling_bert import BertSelfAttention, BertConfig

module = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}

class BiContextEncoder(nn.Module):
	def __init__(self, model, input_dim, output_dim, use_attention=True):
		super(BiContextEncoder, self).__init__()
		# self.lctx_model = ModelFactory.load_model(args.er_model, input_dim, args.er_output_dim)
		self.hidden_size = output_dim
		self.lctx_model = module[model.lower()](input_size=input_dim, hidden_size=self.hidden_size, batch_first=True, bidirectional=False)
		self.rctx_model = module[model.lower()](input_size=input_dim, hidden_size=self.hidden_size, batch_first=True, bidirectional=False)
		self.use_attention = use_attention

	def forward(self, lctx, lctxl, rctx, rctxl):
		lctx = rnn.pack_padded_sequence(lctx, lctxl, batch_first=True, enforce_sorted=False)
		rctx = rnn.pack_padded_sequence(rctx, rctxl, batch_first=True, enforce_sorted=False)

		lctx_emb, lctx_hidden = self.lctx_model(lctx)
		lctx_emb = rnn.pad_packed_sequence(lctx_emb[0], batch_first=True)
		rctx_emb, rctx_hidden = self.rctx_model(rctx)
		rctx_emb = rnn.pad_packed_sequence(rctx_emb[0], batch_first=True)

		if self.use_attention:
			lctx_emb, _ = self.attention(lctx_emb, lctx_hidden)
			rctx_emb, _ = self.attention(rctx_emb, rctx_hidden)

		return F.relu(torch.cat([lctx_emb, rctx_emb], -1))

	def attention(self, lstm_output, final_state):
		# code from https://medium.com/platfarm/%EC%96%B4%ED%85%90%EC%85%98-%EB%A9%94%EC%BB%A4%EB%8B%88%EC%A6%98%EA%B3%BC-transfomer-self-attention-842498fd3225
		hidden = final_state.view(-1, self.hidden_size, 1)
		attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
		soft_attn_weights = F.softmax(attn_weights, 1)
		context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
		return context, soft_attn_weights

class CNNEncoder(nn.Module):
	def __init__(self, in_channel, out_channel, kernel_size):
		super(CNNEncoder, self).__init__()
		self.emb = nn.Sequential(
				nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size),
				nn.MaxPool1d(2)
		)

	def forward(self, input_tensor):
		emb = self.emb(input_tensor)
		return emb.view(input_tensor.size()[0], -1)

class RNNEncoder(nn.Module):
	def __init__(self):
		super(RNNEncoder, self).__init__()

	def forward(self):
		pass

class SelfAttentionEncoder(nn.Module):
	# using https://github.com/huggingface/pytorch-transformers
	def __init__(self, hidden_size, hidden_layers, attention_heads, output_attentions=True):
		super(SelfAttentionEncoder, self).__init__()
		assert hidden_size % attention_heads == 0
		self.config = BertConfig(hidden_size=hidden_size, num_hidden_layers=hidden_layers, num_attention_heads=attention_heads, output_attentions=output_attentions)
		self.encoder = BertSelfAttention(self.config)

	def forward(self, hidden_state, attention_mask=None, head_mask=None):
		if attention_mask is None:
			attention_mask = torch.where(hidden_state != torch.zeros_like(hidden_state), torch.tensor([1.]), torch.tensor([0.]))
		return self.encoder(hidden_state, attention_mask, head_mask)

class Ident(nn.Module):
	def __init__(self, *args, **kwargs):
		super(Ident, self).__init__()

	def forward(self, tensor):
		return tensor

class FFNNEncoder(nn.Module):
	def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
		super(FFNNEncoder, self).__init__()
		layers = [nn.Linear(input_dim, output_dim), nn.ReLU(), nn.Dropout()] if num_layers < 2 else \
			[nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout()] + [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout()] * (num_layers - 2) + [nn.Linear(hidden_dim, output_dim), nn.ReLU(), nn.Dropout()]
		self.nn = torch.nn.Sequential(*layers)

	def forward(self, input_tensor):
		return self.nn(input_tensor)
