from .mulrel_nel import dataset as D
from .mulrel_nel.ed_ranker import EDRanker
from .utils.args import ELArgs
from .utils.data import DataModule
from .utils.postprocess import *
from .. import GlobalValues as gl
from ..ds import Sentence
from ..utils import *
from ..utils import TimeUtil
import torch
class EL:
	def __init__(self, mode="test", model_name="no_okt_ent_emb"):
		gl.logger.info("Initializing EL Module")
		with TimeUtil.TimeChecker("EL_init"):
			if mode == "train":
				self.args = ELArgs()
			else:
				try:
					self.args = ELArgs.from_json("models/el/%s_args.json" % model_name)
				except FileNotFoundError:
					gl.logger.critical("EL %s: No argument file exists!" % model_name)
					import sys
					sys.exit(1)
				except Exception:
					import traceback, sys
					traceback.print_exc()
					sys.exit(1)
			self.data = DataModule(self.args)
			self.model_name = model_name
			self.args.mode = mode
			self.args.model_name = model_name
			self.debug = False
			self.target_device = "cuda" if torch.cuda.is_available() and mode != "demo" else "cpu"
			self.ranker = EDRanker(config=self.config)
			jsondump(self.args.to_json(), "models/el/%s_args.json" % model_name)

	def train(self, train_items, dev_items):
		"""
		Train EL Module
		Input:
			train_items: List of dictionary
			dev_items: List of dictionary
		Output: None
		"""
		gl.logger.debug("Train: %d, Dev: %d" % (len(train_items), len(dev_items)))
		gl.logger.info("Formatting corpus")
		tj, tc, tt = self.data.prepare(*train_items, filter_rate=self.args.train_filter_rate)
		dj, dc, dt = self.data.prepare(*dev_items)
		gl.logger.info("Generating Dataset")
		train_data = D.generate_dataset_from_str(tc, tt)
		dev_data = D.generate_dataset_from_str(dc, dt)
		gl.logger.info("Start training")
		self.ranker.train(train_data, [("dev", dev_data)],
		                  config={'lr': self.args.learning_rate, 'n_epochs': self.args.n_epochs})

	def predict(self, sentences, delete_candidate=True):
		# type_list = [type(x) for x in sentences]
		# assert all([x is str for x in type_list]) or all([x is dict for x in type_list]) or all(
		# 		[x is Sentence for x in type_list])
		batches = split_to_batch(sentences, 100)
		result = []
		jj = []
		ee = {}
		for batch in batches:
			j, conll_str, tsv_str = self.data.prepare(*batch)
			jj += j
			# print(len(batch), len(j))
			dataset = D.generate_dataset_from_str(conll_str, tsv_str)
			data_items = self.ranker.get_data_items(dataset, predict=True)
			# print(len(data_items))
			self.ranker.model._coh_ctx_vecs = []
			predictions = self.ranker.predict(data_items)
			e = D.make_result_dict(dataset, predictions)
			for k, v in ee.items():
				ee[k] = v
			result += merge_item(j, e, delete_candidate)
		if type(sentences[0]) is Sentence:
			result = merge_item_with_corpus(sentences, result)
		if self.debug:
			jsondump(jj, "debug/el_prepare.json")
			jsondump(ee, "debug/el_result_dict.json")
		# print(len(result))
		return result

	def __call__(self, *sentences):
		return self.predict(sentences)

	@property
	def config(self):
		return {
			'hid_dims'           : self.args.hid_dims,
			'emb_dims'           : self.data.entity_embedding.shape[1],
			'freeze_embs'        : True,
			'tok_top_n'          : self.args.tok_top_n,
			'margin'             : self.args.margin,
			'word_voca'          : self.data.word_voca,
			'entity_voca'        : self.data.entity_voca,
			'word_embeddings'    : self.data.word_embedding,
			'entity_embeddings'  : self.data.entity_embedding,
			'snd_word_voca'      : self.data.snd_word_voca,
			'snd_word_embeddings': self.data.snd_word_embedding,
			'dr'                 : self.args.dropout_rate,
			'df'                 : self.args.df,
			'n_loops'            : self.args.n_loops,
			'n_rels'             : self.args.n_rels,
			'mulrel_type'        : self.args.mulrel_type,
			'args'               : self.args,
			'device'             : self.target_device
		}

	def reload_ranker(self):
		self.args.mode = "train"
		self.ranker = EDRanker(config=self.config)
