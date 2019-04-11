
module = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
rnns = [k for k in module.keys()]



class BiContextEREmbedder(nn.Module):
	def __init__(self, args, input_dim):
		super(BiContextEREmbedder, self).__init__()
		# self.lctx_model = ModelFactory.load_model(args.er_model, input_dim, args.er_output_dim)
		self.sequence_len = args.ctx_window_size
		self.lctx_model = module[args.er_model.lower()](input_size=input_dim, hidden_size=args.er_output_dim, batch_first=True, bidirectional=False)
		self.rctx_model = module[args.er_model.lower()](input_size=input_dim, hidden_size=args.er_output_dim, batch_first=True, bidirectional=False)

	def forward(self, lctx, rctx):
		lctx_emb, _ = self.lctx_model(lctx)
		rctx_emb, _ = self.rctx_model(rctx)
		return F.relu(torch.cat([lctx_emb[:, -1, :], rctx_emb[:, -1, :]], -1))

class BiContextELEmbedder(nn.Module):
	def __init__(self, args, input_dim):
		super(BiContextELEmbedder, self).__init__()
		self.lctx_model = module[args.el_model.lower()](input_size=input_dim, hidden_size=args.el_output_dim, batch_first=True, bidirectional=False)
		self.rctx_model = module[args.el_model.lower()](input_size=input_dim, hidden_size=args.el_output_dim, batch_first=True, bidirectional=False)

	def forward(self, lctx, rctx):
		lctx_emb, _ = self.lctx_model(lctx)
		rctx_emb, _ = self.rctx_model(rctx)
		return F.relu(torch.cat([lctx_emb[:, -1, :], rctx_emb[:, -1, :]], -1))
	