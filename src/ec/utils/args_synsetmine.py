import torch
import os
from datetime import datetime

from ...utils import AbstractArgument

class ECArgs(AbstractArgument):
	def __init__(self):
		self.data_format = "set"
		self.modelName = "np_lrlr_sd_lrlrdl"
		self.context_embedder = "rnn"
		self.pretrained_embedding = "embed"
		self.embed_fine_tune = 0
		self.embedSize = 50
		self.node_hiddenSize = 250
		self.combine_hiddenSize = 500
		self.max_set_size = 50

		self.batch_size = 32
		self.lr = 0.0001
		self.loss_fn = "self_margin_rank_bce"
		self.margin = 0.5
		self.epochs = 200
		self.neg_sample_size = 20
		self.neg_sample_method = "share_token"  # one of ["complete_random", "share_token", "mixture"]

		self.dropout = 0.3
		self.early_stop = 100
		self.eval_epoch_step = 5
		self.random_seed = 5417
		self.size_opt_clus = 0
		self.max_K = -1
		self.T = 1
		self.device_id = 0
		self.save_dir = "data/ec/"
		self.load_model = ""
		self.snapshot = ""  # model path
		self.tune_result_file = "tune_prefix"
		self.remark = "ec"

		if self.device_id == -1:
			self.device = torch.device("cpu")
		else:
			os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device_id)
			self.device = torch.device("cuda:0")

		self.comment = ""
		# if self.mode == "train":
		# 	self.comment = '_{}'.format(self.remark)
		# elif self.mode == "tune":
		# 	self.comment = "_{}".format(self.tune_result_file)
		# else:
		# 	self.comment = ""

		# Model snapshot saving
		current_time = datetime.now().strftime('%b%d_%H-%M-%S')
		self.save_dir = os.path.join(self.save_dir, current_time)

		if self.max_K == -1:
			self.max_K = None

		self.size_opt_clus = (self.size_opt_clus == 1)
