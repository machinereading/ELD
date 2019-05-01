from .utils.data import DataModule
from .utils.args import ELArgs
from .utils.postprocess import merge_item
from .mulrel_nel.ed_ranker import EDRanker
from .mulrel_nel import dataset as D
from .. import GlobalValues as gl
from ..utils import TimeUtil
from ..utils import *
from ..ds import Sentence

class EL():
	def __init__(self, mode, model_name):
		gl.logger.info("Initializing EL Module")
		with TimeUtil.TimeChecker("EL_init"):
			self.args = ELArgs()
			self.data = DataModule(self.args)
			self.model_name = model_name
			self.args.mode = mode
			self.args.model_path = "data/el/%s" % model_name
			self.model_name = model_name
			self.debug = False
			
			config={
				'hid_dims': self.args.hid_dims,
				'emb_dims': self.data.entity_embedding.shape[1],
				'freeze_embs': True,
				'tok_top_n': self.args.tok_top_n,
				'margin': self.args.margin,
				'word_voca': self.data.word_voca,
				'entity_voca': self.data.entity_voca,
				'word_embeddings': self.data.word_embedding,
				'entity_embeddings': self.data.entity_embedding,
				'snd_word_voca': self.data.snd_word_voca,
				'snd_word_embeddings': self.data.snd_word_embedding,
				'dr': self.args.dropout_rate,
				'df': self.args.df,
				'n_loops': self.args.n_loops,
				'n_rels': self.args.n_rels,
				'mulrel_type': self.args.mulrel_type,
				'args': self.args
			}
		
			self.ranker = EDRanker(config=self.config)
		

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
		self.ranker.train(train_data, [("dev", dev_data)], config = {'lr': self.args.learning_rate, 'n_epochs': self.args.n_epochs})


	def predict(self, sentences, delete_candidate=True):
		type_list = [type(x) for x in sentences]
		assert all([x is str for x in type_list]) or all([x is dict for x in type_list]) or all([x is Sentence for x in type_list])
		batches = split_to_batch(sentences, 100)
		for batch in batches:
			j, conll_str, tsv_str = self.data.prepare(*batch)
			# print(len(batch), len(j))
			dataset = D.generate_dataset_from_str(conll_str, tsv_str)
			data_items = self.ranker.get_data_items(dataset, predict=True)
			# print(len(data_items))
			self.ranker.model._coh_ctx_vecs = []
			predictions = self.ranker.predict(data_items)
			e = D.make_result_dict(dataset, predictions)
			yield merge_item(j, e, delete_candidate)

	def __call__(self, *sentences):
		result = []
		for batch in self.predict(sentences, "PLAIN_SENTENCE"):
			result += batch
		return result

	@property
	def config(self):
		return {
				'hid_dims': self.args.hid_dims,
				'emb_dims': self.data.entity_embedding.shape[1],
				'freeze_embs': True,
				'tok_top_n': self.args.tok_top_n,
				'margin': self.args.margin,
				'word_voca': self.data.word_voca,
				'entity_voca': self.data.entity_voca,
				'word_embeddings': self.data.word_embedding,
				'entity_embeddings': self.data.entity_embedding,
				'snd_word_voca': self.data.snd_word_voca,
				'snd_word_embeddings': self.data.snd_word_embedding,
				'dr': self.args.dropout_rate,
				'df': self.args.df,
				'n_loops': self.args.n_loops,
				'n_rels': self.args.n_rels,
				'mulrel_type': self.args.mulrel_type,
				'args': self.args
		}
	

	def reload_ranker(self):
		self.args.mode = "train"
		self.ranker = EDRanker(config=self.config)