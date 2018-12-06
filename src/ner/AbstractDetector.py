###################################
# Abstract Mention Detector model #
###################################
from abc import ABC, abstractmethod
from . import NERUtil
from .. import GlobalValues as gl
import tensorflow as tf
class AbstractDetector(ABC):
	def __init__(self):
		self.initialized_variables = False
		self.model_name = gl.ner.model
		self.batch_size = gl.ner.batch_size
		self.label_size = len(gl.ner.label_dict)
		self.highest_eval_score = (-1,-1,-1)
		self.highest_eval_iter = 0

	def train(self, sess, trainset_generator):
		# generate train set and run it!
		for item in self.generate_input(trainset_generator):
			loss, _= sess.run([self.loss, self.optimizer], feed_dict=item)
			yield loss

	def save(self, sess, saver):
		saver.save(sess, gl.target_dir+self.model_name+".ckpt")

	@abstractmethod
	def predict(self, sess, sentences, pos_list=[]):
		#sentence: iterable of iterable tokens
		return None

	@abstractmethod
	def generate_input(self, trainset_generator):
		# generator -> feed_dict
		yield None

	def load_trained_session(self, sess):
		if self.initialized_variables:
			return
		self.saver = tf.train.Saver(tf.global_variables())
		ckpt = tf.train.get_checkpoint_state(gl.target_dir)
		if ckpt:
			self.saver.restore(sess, ckpt.model_checkpoint_path)
			print("Model restored")
		elif gl.run_mode in ["predict", "eval"]:
			raise Exception("No model exists!")
		else:
			print("Initialize variables")
			sess.run(tf.global_variables_initializer())
		self.initialized_variables = True

	def save_trained_session(self, sess, prf, iter=0):
		if prf[-1] > self.highest_eval_score[-1]:
			self.highest_eval_score = prf
			self.highest_eval_iter = iter
			self.saver.save(sess, gl.target_dir+self.model_name+".ckpt")
			print("Model saved")
