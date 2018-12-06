import os

from datetime import datetime
from .BiLSTM_CRF import *
from .CNN_BiLSTM_CRF import *
from . import NEREval
from .. import GlobalValues as gl
from ..utils.datamodule import DataParser
from ..utils import CorpusUtil, printfunc, progress
model_dict = {"BiLSTM_CRF": BiLSTMCRFDetector, "CNN_BiLSTM_CRF": CNNBiLSTMCRFDetector}

class NER():
	def __init__(self, sess):
		self.save_dir = "ner_save/"
		self.sess = sess
		self.epoch = 0
		self.load_result()
		# self.initialize_global_variable()
		data_dir = gl.ner.corpus_dir
		data_generator = getattr(DataParser, gl.ner.corpus_parser)(gl.corpus_home + data_dir)
		print("====Loading corpus====")
		gl.ner.pos_dict, gl.ner.label_dict = CorpusUtil.extract(data_generator.get_trainset())
		gl.ner.pos_dict_inv = {v:k for k,v in gl.ner.pos_dict.items()}
		gl.ner.label_dict_inv = {v:k for k,v in gl.ner.label_dict.items()}

		print("====Corpus loaded====")
		print("====Initializing NER model====")
		if gl.ner.model not in model_dict:
			raise Exception("No such module: %s" % gl.ner.model)
		self.model = model_dict[gl.ner.model]()
		self.model.load_trained_session(self.sess)
		print("====NER model initialized====")
		if gl.run_mode == "train":
			self.train(data_generator)
		if gl.run_mode == "eval":
			self.eval(data_generator)
		if gl.run_mode == "predict":
			self.predict(data_generator)

	# To train, eval, predict with same data form, every input is iterable of string

	def train(self, data_generator):
		print("====Train started====")
		epoch = int(gl.ner.epoch) if hasattr(gl.ner, "epoch") else 50
		eval_iter = int(gl.ner.eval_iter) if hasattr(gl.ner, "eval_iter") else 5
		start = max(list(map(lambda x: int(x), self.train_result.keys()))) if len(self.train_result) > 0 else 0
		self.epoch = start + 1
		for i in range(start+1, start+epoch+1):
			loss_sum = 0
			batch_count = 0
			for loss in self.model.train(self.sess, data_generator.get_trainset()):
				loss_sum += loss
				batch_count += gl.ner.batch_size
				printfunc("%s: Epoch %d %s" % (str(datetime.now()), i, progress(batch_count, min([gl.train_set_count, gl.data.max_train_size]))))
			print()

			if i % eval_iter == 0:
				# self.model.save_trained_session(self.sess, (0,0,0), i)
				p, r, f = self.eval(data_generator)
				self.train_result[i] = {"Precision":p, "Recall":r, "F1": f}
				print("Epoch %d: P %.2f, R %.2f F %.2f" % (i, p, r, f))
				self.model.save_trained_session(self.sess, (p,r,f), i)
				hp, hr, hf = self.model.highest_eval_score
				print("Highest: P %.2f, R %.2f F %.2f @ Epoch %d" % (hp, hr, hf, self.model.highest_eval_iter))
				with open(gl.target_dir+"train_result.json", "w", encoding="UTF8") as f:
					json.dump(self.train_result, f, ensure_ascii=False, indent="\t")
			self.epoch += 1

		print("====Train ended====")



	def eval(self, data_generator, save_result=True):
		eval_set = data_generator.get_devset()
		result_sum = None
		eval_count = 0
		s_batch = []
		p_batch = []
		l_batch = []
		iter_end_flag = False
		if save_result:
			save_file = open(gl.target_dir+gl.ner.corpus_dir.split("/")[0]+"_dev%d" % self.epoch, "w", encoding="UTF8")
			save_file.write("\t".join(["morph", "prediction", "answer"])+"\n")
		while not iter_end_flag and eval_count < 5000:
			try:
				s, p, l = next(eval_set)
			except StopIteration:
				iter_end_flag = True
			s_batch.append(s)
			p_batch.append(p)
			l_batch.append(l)

			if len(s_batch) > gl.ner.batch_size or iter_end_flag:
				# print(s_batch)
				prediction, score = self.model.predict(self.sess, s_batch, p_batch)
				answer_set = [data_generator.decode_label(s, l) for s, l in zip(s_batch, l_batch)]
				prediction_set = [data_generator.decode_label(s, l) for s, l in zip(s_batch, prediction)]
				if save_result:
					for s, p, l in zip(s_batch, prediction, l_batch):
						for m, pp, ll in zip(s, p, l):
							save_file.write("\t".join([m, gl.ner.label_dict_inv[pp], ll])+"\n")
				for a, p in zip(answer_set, prediction_set):
					p = list(filter(lambda x: x["name"] != "" and x["type"] != "", p))
					eval_data = NEREval.compute_metrics(a, p)
					result_sum = NEREval.metric_sum(result_sum, eval_data) if result_sum is not None else eval_data
					eval_count += 1

				s_batch = []
				p_batch = []
				l_batch = []
		tp = result_sum[0]["strict"]["correct"]
		fp = result_sum[0]["strict"]["actual"]
		fn = result_sum[0]["strict"]["possible"]
		if save_result:
			save_file.close()
		return NEREval.prf(tp, fp, fn)


	def predict(self, data_generator, sentence_file=None, wf=None):
		# def batch(sentences):
		# 	sbuf = []
		# 	pbuf = []
		# 	for sentence in sentences:
		# 		if data_generator.pos is not None:
		# 			s = []
		# 			p = []
		# 			for morph, pos in data_generator.pos(sentence):
		# 				s.append(morph)
		# 				p.append(pos)
		# 			sbuf.append(s)
		# 			pbuf.append(p)
		# 		result = self.model.predict(self.sess, [s], [p])
		# 		result = result[0]
		# if sentence_file is not None:
		# 	sbuf = []
		# 	for line in sentence_file.readlines():
		# 		line = line.strip()
		# 		sbuf.append(line)
		# 		if len(sbuf) > 64:
		# 		s = []
		# 		p = []
		# 		if data_generator.pos is not None:
		# 			for morph, pos in data_generator.pos(sentence):
		# 				s.append(morph)
		# 				p.append(pos)
		# 		result = self.model.predict(self.sess, [s], [p])
		# 		result = result[0]
		# 		print(data_generator.decode_label(sentence, result))
		while True:
			sentence = input("문장 입력: ")
			s = []
			p = []
			if data_generator.pos is not None:
				for morph, pos in data_generator.pos(sentence):
					s.append(morph)
					p.append(pos)
			result = self.model.predict(self.sess, [s], [p])
			result = result[0]
			print(data_generator.decode_label(sentence, result))


	def load_result(self):
		if not os.path.isfile(gl.target_dir+"train_result.json"):
			self.train_result = {}
			return
		with open(gl.target_dir+"train_result.json", encoding="UTF8") as f:
			self.train_result = json.load(f)


	# def initialize_global_variable(self):
	# 	gl.batch_maximum = getattr(gl.ner, "batch_maximum", -1)
	# 	gl.batch_size = getattr(gl, "batch_size", 64)