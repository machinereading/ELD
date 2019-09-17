from .utils.args import ERArgs
from .utils.data import sentence2conll
from .NeuroNLP2.io import get_logger, conll03_data, CoNLL03Writer
from .NeuroNLP2.models import BiRecurrentConvCRF, BiVarRecurrentConvCRF

from .NeuroNLP2 import utils
from ..utils import TimeUtil

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD

import os

def evaluate(output_file, model_name, epoch):
	score_file = "runs/er/score_%s_%d" % (model_name, epoch)
	os.system("src/er/NeuroNLP2/examples/eval/conll03eval.v2 < %s > %s" % (output_file, score_file))
	with open(score_file, 'r') as fin:
		fin.readline()
		line = fin.readline()
		fields = line.split(";")
		acc = float(fields[0].split(":")[1].strip()[:-1])
		precision = float(fields[1].split(":")[1].strip()[:-1])
		recall = float(fields[2].split(":")[1].strip()[:-1])
		f1 = float(fields[3].split(":")[1].strip())
	return acc, precision, recall, f1

class ER:
	def __init__(self, model_name):
		self.args = ERArgs()
		self.model_name = model_name
		self.use_gpu = torch.cuda.is_available()
		self.embedd_dict, self.embedd_dim = utils.load_embedding_dict(self.args.embedding, self.args.embedding_dict)
		self.load_model()
		
		


	def train(self, train_corpus_path, dev_corpus_path):
		word_alphabet, char_alphabet, pos_alphabet, \
		chunk_alphabet, ner_alphabet = conll03_data.create_alphabets("data/er/alphabets/", train_corpus_path, data_paths=[dev_corpus_path],
																 embedd_dict=self.embedd_dict, max_vocabulary_size=50000)
		data_train = conll03_data.read_data_to_variable(train_corpus_path, word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet, use_gpu=self.use_gpu)
		data_dev = conll03_data.read_data_to_variable(dev_corpus_path, word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet, use_gpu=self.use_gpu, volatile=True)
		writer = CoNLL03Writer(word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet)

		char_dim = self.args.char_dim
		window = 3
		num_layers = self.args.num_layers
		tag_space = self.args.tag_space
		initializer = nn.init.xavier_uniform
		num_data = sum(data_train[1])
		num_labels = ner_alphabet.size()
		if self.network is None:
			self.network = BiRecurrentConvCRF(self.embedd_dim, word_alphabet.size(), char_dim, char_alphabet.size(), self.args.num_filters, window, self.args.mode, self.args.hidden_size, num_layers, num_labels,
									 tag_space=tag_space, embedd_word=self.word_table, p_in=self.args.p_in, p_out=self.args.p_out, p_rnn=tuple(self.args.p_rnn), bigram=self.args.bigram, initializer=initializer)
		if self.use_gpu:
			self.network.cuda()
		lr = self.args.learning_rate
		optim = SGD(self.network.parameters(), lr=lr, momentum=0.9, weight_decay=self.args.gamma, nesterov=True)
		num_batches = num_data // self.args.batch_size + 1
		dev_f1 = 0.0
		with TimeUtil.TimeChecker("NER Training"):
			for epoch in tqdm(range(self.args.num_epochs), desc="NER Training", initial=self.start_epoch):
				train_err = 0.
				train_total = 0.
				self.network.train()
				for batch in range(1, num_batches + 1):
					if batch % 10 != 0: continue # limit train set to test workability
					word, char, _, _, labels, masks, lengths = conll03_data.get_batch_variable(data_train, self.args.batch_size, unk_replace=self.args.unk_replace)
					optim.zero_grad()
					loss = self.network.loss(word, char, labels, mask=masks)
					loss.backward()
					optim.step()

					num_inst = word.size(0)
					train_err += loss.data[0] * num_inst
					train_total += num_inst

				self.network.eval()
				fname = "runs/er/"+self.model_name+str(epoch)
				writer.start(fname)
				for batch in conll03_data.iterate_batch_variable(data_dev, self.args.batch_size):
					word, char, pos, chunk, labels, masks, lengths = batch
					preds, _ = self.network.decode(word, char, target=labels, mask=masks, leading_symbolic=conll03_data.NUM_SYMBOLIC_TAGS)
					writer.write(word.data.cpu().numpy(), pos.data.cpu().numpy(), chunk.data.cpu().numpy(), preds.cpu().numpy(), labels.data.cpu().numpy(), lengths.cpu().numpy())
				writer.close()
				acc, precision, recall, f1 = evaluate(fname, self.model_name, epoch)
				if dev_f1 < f1:
					dev_f1 = f1
					torch.save(self.network.state_dict(), "models/er/%s_%d.pt" % (self.model_name, epoch))


	def predict(self, sentence):
		word_alphabet, char_alphabet, pos_alphabet, \
		chunk_alphabet, ner_alphabet = conll03_data.load_alphabets("data/er/alphabets/")
		writer = CoNLL03Writer(word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet)
		self.network.eval()
		conll_form = sentence2conll(sentence)
		preds, _ = self.network.decode(word, char, target=labels, mask=masks, leading_symbolic=conll03_data.NUM_SYMBOLIC_TAGS)
		writer.write(word.data.cpu().numpy(), pos.data.cpu().numpy(), chunk.data.cpu().numpy(), preds.cpu().numpy(), labels.data.cpu().numpy(), lengths.cpu().numpy())
		writer.close()
		return preds

	def __call__(self, *sentence):
		return self.predict(sentence)

	def construct_word_embedding_table(self, word_alphabet):
		scale = np.sqrt(3.0 / self.embedd_dim)
		table = np.empty([word_alphabet.size(), self.embedd_dim], dtype=np.float32)
		table[conll03_data.UNK_ID, :] = np.random.uniform(-scale, scale, [1, self.embedd_dim]).astype(np.float32)
		oov = 0
		for word, index in word_alphabet.items():
			if word in self.embedd_dict:
				embedding = self.embedd_dict[word]
			elif word.lower() in self.embedd_dict:
				embedding = self.embedd_dict[word.lower()]
			else:
				embedding = np.random.uniform(-scale, scale, [1, self.embedd_dim]).astype(np.float32)
				oov += 1
			table[index, :] = embedding
		print('oov: %d' % oov)
		return torch.from_numpy(table)

	def load_model(self):
		word_alphabet, char_alphabet, pos_alphabet, \
		chunk_alphabet, ner_alphabet = conll03_data.load_alphabets("data/er/alphabets/")
		word_table = self.construct_word_embedding_table(word_alphabet)
		initializer = nn.init.xavier_uniform_
		self.network = BiRecurrentConvCRF(self.embedd_dim, word_alphabet.size(), self.args.char_dim, char_alphabet.size(), self.args.num_filters, 3, self.args.mode, self.args.hidden_size, self.args.num_layers, ner_alphabet.size(),
									 tag_space=self.args.tag_space, embedd_word=word_table, p_in=self.args.p_in, p_out=self.args.p_out, p_rnn=tuple(self.args.p_rnn), bigram=self.args.bigram, initializer=initializer)
		max_epoch = 0
		for fname in os.listdir("models/er/"):
			try:
				model_name, epoch = fname.split("_")
				if model_name != self.model_name:
					continue
				if int(epoch) > max_epoch:
					max_epoch = int(epoch)
			except:
				pass
		if max_epoch > 0:
			self.network.load_state_dict(torch.load("models/er/%s_%d.pt" % (self.model_name, max_epoch)) if max_epoch > 0 else None)
		self.start_epoch = max_epoch
