import torch
import torch.nn as nn
import torch.nn.functional as F

module = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}

class BiContextEmbedder(nn.Module):
	def __init__(self, model, input_dim, output_dim):
		super(BiContextEmbedder, self).__init__()
		# self.lctx_model = ModelFactory.load_model(args.er_model, input_dim, args.er_output_dim)
		self.lctx_model = module[model.lower()](input_size=input_dim, hidden_size=output_dim, batch_first=True, bidirectional=False)
		self.rctx_model = module[model.lower()](input_size=input_dim, hidden_size=output_dim, batch_first=True, bidirectional=False)

	def forward(self, lctx, rctx):
		lctx_emb, _ = self.lctx_model(lctx)
		rctx_emb, _ = self.rctx_model(rctx)
		# return F.relu(lctx_emb), F.relu(rctx_emb)
		return F.relu(torch.cat([lctx_emb[:, -1, :], rctx_emb[:, -1, :]], -1))

class CNNEmbedder(nn.Module):
	def __init__(self, in_channel, out_channel, kernel_size):
		super(CNNEmbedder, self).__init__()
		self.emb = nn.Sequential(
				nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size),
				nn.MaxPool1d(2)
		)

	def forward(self, input_tensor):
		emb = self.emb(input_tensor)
		return emb.view(input_tensor.size()[0], -1)
