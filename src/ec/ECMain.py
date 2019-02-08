from .utils.args import EC_Args
from .utils.el_dataset_merger import generate_data_from_el_result
from .synsetmine.model import SSPM
from .synsetmine.dataloader import element_set
from .synsetmine import cluster_predict
from .synsetmine import evaluator
from .synsetmine.utils import save_model, load_model, my_logger, load_embedding, load_raw_data, Results, Metrics

from ..utils import readfile, TimeUtil

from tqdm import tqdm
import torch
import random
import numpy as np

class EC():
	def __init__(self):
		self.args = EC_Args()
		self.options = vars(self.args)
		self.debug = False
		random.seed(self.args.random_seed)
		torch.manual_seed(self.args.random_seed)
		np.random.seed(self.args.random_seed)
		if self.args.device_id != -1:
			torch.cuda.manual_seed_all(self.args.random_seed)
			torch.backends.cudnn.deterministic = True
		torch.set_printoptions(precision=9)
		torch.set_num_threads(1)
		fi = "data/ec/ec_embedding.w2v"
		embedding, index2word, word2index, vocab_size, embed_dim = load_embedding(fi)
		# logger.info("Finish loading embedding: embed_dim = {}, vocab_size = {}".format(embed_dim, vocab_size))
		self.options["embedding"] = embedding
		self.options["index2word"] = index2word
		self.options["word2index"] = word2index
		self.options["vocabSize"] = vocab_size

	def train(self, train_corpus, dev_corpus=None):
		self.args.mode = "train"
		self.options["mode"] = "train"
		if type(train_corpus) is str:
			f = [x for x in readfile(train_corpus)]
		else:
			f = []
			for line in train_corpus.readlines():
				f.append(line.strip())
		if dev_corpus is not None:
			if type(train_corpus) is str:
				df = readfile(train_corpus)
			else:
				df = []
				for line in train_corpus.readlines():
					df.append(line.strip())
			dev_set = element_set.ElementSet("dev_set", self.options["data_format"], self.options, f)
		else:
			dev_set = None
		random.shuffle(f)
		train_set = element_set.ElementSet("train_set", self.options["data_format"], self.options, f)
		train_results = Results("log/ec/log.txt") # does it need to write on something?
		options = self.options
		if self.debug:
			# Add TensorBoard Writer
			tb_writer = SummaryWriter(log_dir=None, comment=args.comment)

			# Add Python Logger
			my_logger = my_logger(name='exp', log_path=writer.file_writer.get_logdir())
			my_logger.setLevel(0)
		# synsetmine.main.run
		model = SSPM(options)
		model = model.to(options["device"])
		optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=options["lr"], amsgrad=True)
		results = Metrics()
		# Training phase
		train_set._shuffle()
		train_set_size = len(train_set)
		print("train_set_size: {}".format(train_set_size))

		model.train()
		early_stop_metric_name = "FMI"  # metric used for early stop
		best_early_stop_metric = 0.0
		last_step = 0
		save_model(model, options["save_dir"], 'best', 0)  # save the initial first model

		for epoch in tqdm(range(options["epochs"]), desc="Training ..."):
			loss = 0
			epoch_samples = 0
			epoch_tn = 0
			epoch_fp = 0
			epoch_fn = 0
			epoch_tp = 0
			for train_batch in train_set.get_train_batch(max_set_size=options["max_set_size"],
														 neg_sample_size=options["neg_sample_size"],
														 neg_sample_method=options["neg_sample_method"],
														 batch_size=options["batch_size"]):
				train_batch["data_format"] = "sip"
				optimizer.zero_grad()
				cur_loss, tn, fp, fn, tp = model.train_step(train_batch)
				optimizer.step()

				loss += cur_loss
				epoch_tn += tn
				epoch_fp += fp
				epoch_fn += fn
				epoch_tp += tp
				epoch_samples += (tn + fp + fn + tp)

				epoch_precision, epoch_recall, epoch_f1 = evaluator.calculate_precision_recall_f1(tp=epoch_tp, fp=epoch_fp,
																								  fn=epoch_fn)
			epoch_accuracy = 1.0 * (epoch_tp + epoch_tn) / epoch_samples
			loss /= epoch_samples
			my_logger.info("    train/loss (per instance): {}".format(loss))
			my_logger.info("    train/precision: {}".format(epoch_precision))
			my_logger.info("    train/recall: {}".format(epoch_recall))
			my_logger.info("    train/accuracy: {}".format(epoch_accuracy))
			my_logger.info("    train/f1: {}".format(epoch_f1))
			tb_writer.add_scalar('train/loss (per instance)', loss, epoch)
			tb_writer.add_scalar('train/precision', epoch_precision, epoch)
			tb_writer.add_scalar('train/recall', epoch_recall, epoch)
			tb_writer.add_scalar('train/accuracy', epoch_accuracy, epoch)
			tb_writer.add_scalar('train/f1', epoch_f1, epoch)

			if epoch % options["eval_epoch_step"] == 0 and epoch != 0:
				# set-instance pair prediction evaluation
				metrics = evaluator.evaluate_set_instance_prediction(model, dev_set)
				tb_writer.add_scalar('val-sip/sip-precision', metrics["precision"], epoch)
				tb_writer.add_scalar('val-sip/sip-recall', metrics["recall"], epoch)
				tb_writer.add_scalar('val-sip/sip-f1', metrics["f1"], epoch)
				tb_writer.add_scalar('val-sip/sip-loss', metrics["loss"], epoch)
				my_logger.info("    val/sip-precision: {}".format(metrics["precision"]))
				my_logger.info("    val/sip-recall: {}".format(metrics["recall"]))
				my_logger.info("    val/sip-f1: {}".format(metrics["f1"]))
				my_logger.info("    val/sip-loss: {}".format(metrics["loss"]))

				# clustering evaluation
				vocab = dev_set.vocab
				cls_pred = cluster_predict.set_generation(model, vocab, size_opt_clus=options["size_opt_clus"],
														  max_K=options["max_K"])
				cls_true = dev_set.positive_sets
				metrics_cls = evaluator.evaluate_clustering(cls_pred, cls_true)
				tb_writer.add_scalar('val-cluster/ARI', metrics_cls["ARI"], epoch)
				tb_writer.add_scalar('val-cluster/FMI', metrics_cls["FMI"], epoch)
				tb_writer.add_scalar('val-cluster/NMI', metrics_cls["NMI"], epoch)
				tb_writer.add_scalar('val-cluster/em', metrics_cls["num_of_exact_set_prediction"], epoch)
				tb_writer.add_scalar('val-cluster/mwm_jaccard', metrics_cls["maximum_weighted_match_jaccard"], epoch)
				tb_writer.add_scalar('val-cluster/inst_precision', metrics_cls["pair_precision"], epoch)
				tb_writer.add_scalar('val-cluster/inst_recall', metrics_cls["pair_recall"], epoch)
				tb_writer.add_scalar('val-cluster/inst_f1', metrics_cls["pair_f1"], epoch)
				tb_writer.add_scalar('val-cluster/cluster_num', metrics_cls["num_of_predicted_clusters"], epoch)
				my_logger.info("    val/ARI: {}".format(metrics_cls["ARI"]))
				my_logger.info("    val/FMI: {}".format(metrics_cls["FMI"]))
				my_logger.info("    val/NMI: {}".format(metrics_cls["NMI"]))
				my_logger.info("    val/em: {}".format(metrics_cls["num_of_exact_set_prediction"]))
				my_logger.info("    val/mwm_jaccard: {}".format(metrics_cls["maximum_weighted_match_jaccard"]))
				my_logger.info("    val/inst_precision: {}".format(metrics_cls["pair_precision"]))
				my_logger.info("    val/inst_recall: {}".format(metrics_cls["pair_recall"]))
				my_logger.info("    val/inst_f1: {}".format(metrics_cls["pair_f1"]))
				my_logger.info("    val/cluster_num: {}".format(metrics_cls["num_of_predicted_clusters"]))
				my_logger.info("    val/clus_size2num_pred_clus: {}".format(metrics_cls["cluster_size2num_of_predicted_clusters"]))

				# Early stop based on clustering results
				if metrics_cls[early_stop_metric_name] > best_early_stop_metric:
					best_early_stop_metric = metrics_cls[early_stop_metric_name]
					last_step = epoch
					save_model(model, options["save_dir"], 'best', epoch)
				my_logger.info("-" * 80)

			if epoch - last_step > options["early_stop"]:
				print("Early stop by {} steps, best {}: {}, best step: {}".format(epoch, early_stop_metric_name,
																				  best_early_stop_metric, last_step))
				break

			train_set._shuffle()

		my_logger.info("Final Results:")
		my_logger.info("Loading model: {}/best_steps_{}.pt".format(options["save_dir"], last_step))
		load_model(model, options["save_dir"], 'best', last_step)
		model = model.to(options["device"])

		my_logger.info("=== Set-Instance Prediction Metrics ===")
		metrics = evaluator.evaluate_set_instance_prediction(model, dev_set)
		for metric in metrics:
			my_logger.info("    {}: {}".format(metric, metrics[metric]))

		my_logger.info("=== Clustering Metrics ===")
		vocab = dev_set.vocab
		cls_pred = cluster_predict.set_generation(model, vocab, size_opt_clus=options["size_opt_clus"],
												  max_K=options["max_K"])
		cls_true = dev_set.positive_sets
		metrics_cls = evaluator.evaluate_clustering(cls_pred, cls_true)
		for metric in metrics_cls:
			if not isinstance(metrics_cls[metric], list):
				my_logger.info("    {}: {}".format(metric, metrics_cls[metric]))

		# save all metrics
		results.add("sip-f1", metrics["f1"])
		results.add("sip-precision", metrics["precision"])
		results.add("sip-recall", metrics["recall"])
		results.add("ARI", metrics_cls["ARI"])
		results.add("FMI", metrics_cls["FMI"])
		results.add("NMI", metrics_cls["NMI"])
		results.add("pred_clus_num", metrics_cls["num_of_predicted_clusters"])
		results.add("em", metrics_cls["num_of_exact_set_prediction"])
		results.add("mwm_jaccard", metrics_cls["maximum_weighted_match_jaccard"])
		results.add("inst-precision", metrics_cls["pair_precision"])
		results.add("inst-recall", metrics_cls["pair_recall"])
		results.add("inst-f1", metrics_cls["pair_f1"])

		interested_hyperparameters = ["modelName", "dataset", "data_format", "pretrained_embedding", "embedSize",
									  "node_hiddenSize", "combine_hiddenSize", "batch_size", "neg_sample_size", "lr",
									  "dropout", "early_stop", "random_seed", "save_dir"]
		hyperparameters = {}
		for hyperparameter_name in interested_hyperparameters:
			hyperparameters[hyperparameter_name] = options[hyperparameter_name]
		train_results.save_metrics(hyperparameters, metrics)

	@TimeUtil.measure_time
	def cluster(self, corpus):
		self.args.mode = "cluster_predict"
		self.options["mode"] = "cluster_predict"
		if type(corpus) is str:
			f = readfile(corpus)
		else:
			f = []
			for line in corpus.readlines():
				f.append(line.strip())

		test_set = element_set.ElementSet("test_set", "set", self.options, f)
		
		model = SSPM(self.options)
		model = model.to(self.options["device"])
		model_path = "data/ec/wiki_train.pt"
		model.load_state_dict(torch.load(model_path))
		vocab = test_set.vocab
		clusters = cluster_predict.set_generation(model, vocab, threshold=0.5, eid2ename=test_set.index2word)
		return clusters

	def __call__(self, el_result):
		input_data = generate_data_from_el_result(el_result)
		return self.cluster(input_data)