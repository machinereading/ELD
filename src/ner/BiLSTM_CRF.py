# simple bilstm-crf model based on morpheme

import tensorflow as tf
import numpy as np
from .AbstractDetector import AbstractDetector
from ..utils import LazyProperty as LP
from .. import GlobalValues as gl
class BiLSTMCRFDetector(AbstractDetector):
	def __init__(self):
		super().__init__()
		self.placeholder_size = gl.embedding.dim # temp
		self.learning_rate = getattr(gl, "learning_rate", 0.01)
		self.decay_rate = getattr(gl, "decay_rate", 0.99)
		self.use_pos = gl.boolmap[getattr(gl.ner, "use_pos", "False")]

		if gl.boolmap[getattr(gl.ner, "use_pos", "False")]:
			self.placeholder_size += len(gl.ner.pos_dict)
		hidden_layer_size = int(gl.ner.layer_size)
		with tf.name_scope("LSTM_CRF"):
			self.sentence_embedding = tf.placeholder(tf.float32, [None, None, self.placeholder_size])
			self.sentence_length = tf.placeholder(tf.int32, [None, ]) # length of sentence -> CHAR LENGTH!!!
			self.labels = tf.placeholder(tf.int32, [None, None])
			
			cell_caller = {"RNN": tf.nn.rnn_cell.BasicRNNCell, "LSTM": tf.nn.rnn_cell.LSTMCell, "GRU": tf.nn.rnn_cell.GRUCell}
			# cell_caller = tf.contrib.cudnn_rnn.CudnnLSTM
			# BiLSTM Layer
			# cell_fw = cell_caller(num_layers=int(gl.num_layers), num_units=hidden_layer_size)
			# cell_bw = cell_caller(num_layers=int(gl.num_layers), num_units=hidden_layer_size)

			cell_fw = tf.contrib.rnn.MultiRNNCell([cell_caller[gl.ner.cell](hidden_layer_size, name="LSTM%d" % l) for l in range(int(gl.ner.num_layers))])
			cell_bw = tf.contrib.rnn.MultiRNNCell([cell_caller[gl.ner.cell](hidden_layer_size, name="LSTM%d" % l) for l in range(int(gl.ner.num_layers))])

			name_index = 0
			(output_fw, output_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.sentence_embedding, sequence_length=self.sentence_length, dtype=tf.float32)

			# CRF Layer
			# context_rep = tf.layers.dropout(tf.concat([output_fw, output_bw], axis=-1), rate=gl.ner.dropout_rate)
			context_rep = tf.concat([output_fw, output_bw], axis=-1)


			ntime_steps = tf.shape(context_rep)[1]
			context_rep_flat = tf.reshape(context_rep, [-1, 2*hidden_layer_size])
			w = tf.get_variable("W", shape=[2*hidden_layer_size, self.label_size], dtype=tf.float32)
			b = tf.get_variable("b", shape=[self.label_size], dtype=tf.float32)
			pred = tf.layers.dropout(tf.matmul(context_rep_flat, w)+b, rate=0.4)
			# pred = tf.matmul(context_rep_flat, w)+b
			self.scores = tf.reshape(pred, [-1, ntime_steps, self.label_size])# BATCH, ?, 5
			print(self.scores.shape)
			self.log_likelihood, self.transition_matrix = tf.contrib.crf.crf_log_likelihood(self.scores, self.labels, self.sentence_length)
			# train
			softmax_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=tf.one_hot(self.labels, self.label_size)))
			self.loss = tf.reduce_mean(-self.log_likelihood)
			
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
			self.result_seq, self.seq_score = tf.contrib.crf.crf_decode(self.scores, self.transition_matrix, self.sentence_length)
			self.binary_score = tf.contrib.crf.crf_binary_score(self.result_seq, self.sentence_length, self.transition_matrix)

	def predict(self, sess, sentences, pos_list=[]):
		if len(sentences) == 0: return [], None
		len_list = list(map(len, sentences))
		maxlen = max(len_list)
		feed = {self.sentence_embedding: np.array(list(map(lambda x: self.embedding_generator(x, maxlen), sentences))), self.sentence_length: len_list}
		a, b = sess.run([self.result_seq, self.binary_score], feed_dict=feed)
		return a, b

	def generate_input(self, trainset):
		c = 0
		while True:
			if gl.data.max_train_size > -1 and c > gl.data.max_train_size:
				return
			svbuf = []
			lbuf = []
			lens = []
			posbuf = []
			t = 0
			while t < gl.ner.batch_size:
				try:
					sentence, pos, label = next(trainset)
					if len(sentence) > 250 or len(sentence) < 10: continue # 너무 길거나 짧은 문장 제외 - 메모리 터짐
					if np.random.rand() < gl.data.train_data_drop_rate: continue # random sampling
					if all(map(lambda x: x == "O", label)): continue # 최소 1개의 entity가 있는 것만
					c += 1
					t += 1
				except StopIteration:
					return
				svbuf.append(sentence)
				lens.append(len(sentence))
				lbuf.append(label) # padding
				posbuf.append(pos)
			maxlen = max(list(map(len, svbuf)))
			# if gl.use_char_embedding == "True":
			# 	embedding_func = self.generate_concatenated_embedding_from_sentence
			# else:

			# x = np.array(list(map(lambda x: self.embedding_generator(x, maxlen), svbuf)))
			x = np.array([self.embedding_generator(x, maxlen, y) for x, y in zip(svbuf, posbuf)])
			lbuf = list(map(lambda x: x + ["O"] * (maxlen - len(x)), lbuf)) # 0으로 tagging하기 위해
			label_num = np.array([list(map(gl.ner.label_dict.__getitem__, item)) for item in lbuf])
			# print(label_num)
			lens = np.array(lens)
			# print(x.shape, label_num.shape)
			if len(x.shape) != 3 or len(label_num.shape) != 2: # something went wrong when making input vectors, throw whole set
				c -= gl.ner.batch_size
				continue
			self.learning_rate *= self.decay_rate
			# print(x.shape)
			yield {
					self.sentence_embedding: x, # batch_size, max_sentence_length, word_embedding
					self.sentence_length: lens, # batch_size
					self.labels: label_num      # batch_size, max_sentence_length
				  }

	def embedding_generator(self, morphs, maxlen, pos=None):
		# temporary
		result = []
		if pos is None:
			pos = [None] * len(morphs)
		else:
			pos = [gl.ner.pos_dict[p] for p in pos]
		for morph, pos in zip(morphs, pos):
			embedding = gl.embedding[morph]
			if self.use_pos:
				embedding = np.concatenate([embedding, gl.one_hot(pos if pos is not None else 0, len(gl.ner.pos_dict))], -1)
			result.append(embedding)

		while len(result) < maxlen:
			result.append(np.zeros([self.placeholder_size,], np.float32))
		# print(np.array(result).shape)
		return result