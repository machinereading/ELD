import os
import time

import tensorflow as tf

from .CR import util
from .CR import coref_model
from .utils import DataModule
from .. import GlobalValues as gl
from ..utils import jsondump
class EC:
	def __init__(self, model_name):
		gl.logger.info("Initializing EC Model")
		self.config = util.initialize_from_env(model_name)
		self.model_name = model_name
		self.datamodule = DataModule()
		self.model = coref_model.CorefModel(self.config)
		self.session = tf.Session()
		try:
			self.model.restore(self.session)
		except Exception:
			pass

	def train(self, data):
		gl.logger.info("EC Training")
		report_frequency = self.config["report_frequency"]
		eval_frequency = self.config["eval_frequency"]
		saver = tf.train.Saver()
		dataset, etri = self.datamodule.generate_training_data(data)
		train_dataset = dataset[:int(len(dataset) * 0.9)]
		dev_dataset = dataset[int(len(dataset) * 0.9):]
		log_dir = self.config["log_dir"]
		writer = tf.summary.FileWriter(log_dir, flush_secs=20)

		max_f1 = 0

		# tf_config = tf.ConfigProto(device_count = {'GPU': 1})
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

		with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
			# with tf.Session() as session:
			session.run(tf.global_variables_initializer())
			self.model.start_enqueue_thread(session, train_dataset)
			accumulated_loss = 0.0

			ckpt = tf.train.get_checkpoint_state(log_dir)
			if ckpt and ckpt.model_checkpoint_path:
				print("Restoring from: {}".format(ckpt.model_checkpoint_path))
				saver.restore(session, ckpt.model_checkpoint_path)

			initial_time = time.time()
			while True:
				tf_predictions, tf_loss, tf_global_step, _ = session.run(
						[self.model.predictions, self.model.loss, self.model.global_step, self.model.train_op])
				accumulated_loss += tf_loss
				# print(tf_predictions[2])
				if tf_global_step % report_frequency == 0:
					total_time = time.time() - initial_time
					steps_per_second = tf_global_step / total_time
					average_loss = accumulated_loss / report_frequency
					print("[{}] loss={:.2f}, steps/s={:.2f}".format(tf_global_step, average_loss, steps_per_second))
					writer.add_summary(util.make_summary({"loss": average_loss}), tf_global_step)
					accumulated_loss = 0.0

				# print(session.run([model.textcnn_scores, model.candidate_mention,score]))

				if tf_global_step % eval_frequency == 0:
					saver.save(session, os.path.join(log_dir, "model"), global_step=tf_global_step)
					eval_summary, eval_f1 = self.model.evaluate(session, dev_dataset)

					if eval_f1 >= max_f1:
						max_f1 = eval_f1
						util.copy_checkpoint(os.path.join(log_dir, "model-{}".format(tf_global_step)),
						                     os.path.join(log_dir, "model.max.ckpt"))

					writer.add_summary(eval_summary, tf_global_step)
					writer.add_summary(util.make_summary({"max_eval_f1": max_f1}), tf_global_step)

					print("[{}] evaL_f1={:.2f}, max_f1={:.2f}".format(tf_global_step, eval_f1, max_f1))

				if tf_global_step == 30001:
					break

	def __call__(self, el_data):
		return self.predict_coreference(el_data)

	def predict_coreference(self, el_input):
		jsonline_input, etri = self.datamodule.convert_data(el_input)
		prediction = []
		for example_num, example in enumerate(jsonline_input):
			tensorized_example = self.model.tensorize_example(example, is_training=False)
			feed_dict = {i: t for i, t in zip(self.model.input_tensors, tensorized_example)}
			_, _, _, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = session.run(
					self.model.predictions, feed_dict=feed_dict)
			predicted_antecedents = self.model.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
			example["predicted_clusters"], _ = self.model.get_predicted_clusters(top_span_starts, top_span_ends,
			                                                                     predicted_antecedents)
			prediction.append(example)
		return self.datamodule.postprocess(jsonline_input, etri, prediction)
