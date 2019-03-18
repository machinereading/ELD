import torch
import torch.nn as nn



def load_model(model_name, input_dim, output_dim, *, layers=1, dropout_rate=0.4, bidirectional=True):
	if model_name == "CNN":
		# input: Batch * Channel_in * 
		# return nn.Conv1d()
		return None
	elif model_name == "RNN":
		return nn.RNN(input_size=input_dim, hidden_size=output_dim, num_layers=layers, dropout=dropout_rate, bidirectional=bidirectional)
	elif model_name == "LSTM":
		return nn.LSTM(input_size=input_dim, hidden_size=output_dim, bidirectional=bidirectional)
	elif model_name == "GRU":
		return nn.GRU(input_size=input_dim, hidden_size=output_dim, num_layers=layers, dropout=dropout_rate, bidirectional=bidirectional)
	elif model_name == "FFNN":
		return nn.Sequential(
			nn.Linear(input_dim, output_dim),
			nn.ReLU(),
			nn.Dropout(dropout)
		)