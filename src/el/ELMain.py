from .utils import data as data
from .utils import args
from .utils.postprocess import merge_item
from .mulrel_nel.ed_ranker import EDRanker
from .mulrel_nel import dataset as D
from .mulrel_nel import utils as U
from .. import GlobalValues as gl
from ..utils import TimeUtil
from ..utils import jsondump
class EL():
	def __init__(self, mode, model_name):
		with TimeUtil.TimeChecker("EL_init"):
			self.arg = args.get_args()
			self.arg.mode = mode
			self.arg.model_path = "data/el/%s" % model_name
			self.debug = True
			arg = self.arg
			voca_emb_dir = 'data/el/embeddings/'

			word_voca, word_embeddings = U.load_voca_embs(voca_emb_dir + 'dict.word',
														  voca_emb_dir + 'word_embeddings.npy')
			snd_word_voca, snd_word_embeddings = U.load_voca_embs(voca_emb_dir + '/glove/dict.word',
															  voca_emb_dir + '/glove/word_embeddings.npy')
			entity_voca, entity_embeddings = U.load_voca_embs(voca_emb_dir + 'dict.entity',
															  voca_emb_dir + 'entity_embeddings.npy')
			
			config={
				'hid_dims': arg.hid_dims,
				'emb_dims': entity_embeddings.shape[1],
				'freeze_embs': True,
				'tok_top_n': arg.tok_top_n,
				'margin': arg.margin,
				'word_voca': word_voca,
				'entity_voca': entity_voca,
				'word_embeddings': word_embeddings,
				'entity_embeddings': entity_embeddings,
				'snd_word_voca': snd_word_voca,
				'snd_word_embeddings': snd_word_embeddings,
				'dr': arg.dropout_rate,
				'args': arg
			}
			config['df'] = arg.df
			config['n_loops'] = arg.n_loops
			config['n_rels'] = arg.n_rels
			config['mulrel_type'] = arg.mulrel_type
		
			self.ranker = EDRanker(config=config)
		

	def train(self, train_items, dev_items):
		"""
		Train EL Module
		Input:
			train_items: List of dictionary
			dev_items: List of dictionary
		Output: None
		"""
		# tj = []
		# dj = []
		# for item in train_items:
		# 	j, c, t = data.prepare_sentence(item, ne_marked=True)
		# 	tj.append(j)

		print("TRAINSET")
		tj, tc, tt = data.prepare(*train_items, form="ETRI")
		print(len(train_items), len(tj))
		# for item in train_items:
		# 	try:
		# 		j, c, t = data.prepare_sentence(item, ne_marked=True)
		# 		tj.append(j)
		# 		tc += c + [""]
		# 		tt += t
		# 	except:
		# 		pass

		print("DEVSET")
		dj, dc, dt = data.prepare(*dev_items, form="ETRI")
		# for item in dev_items:
		# 	try:
		# 		j, c, t = data.prepare_sentence(item, ne_marked=True)
		# 		dj.append(j)
		# 		dc += c + [""]
		# 		dt += t
		# 	except:
		# 		pass
		print(len(dev_items), len(dj))
		with open("tc", "w", encoding="UTF8") as f:
			for line in tc:
				f.write(line+"\n")
		with open("tt", "w", encoding="UTF8") as f:
			for line in tt:
				f.write(line+"\n")
		with open("dc", "w", encoding="UTF8") as f:
			for line in dc:
				f.write(line+"\n")
		with open("dt", "w", encoding="UTF8") as f:
			for line in dt:
				f.write(line+"\n")
		# tj, tc, tt = data.prepare(*train_items, is_json=True)
		# dj, dc, dt = data.prepare(*dev_items, is_json=True)
		train_data = D.generate_dataset_from_str(tc, tt)
		dev_data = D.generate_dataset_from_str(dc, dt)
		self.ranker.train(train_data, [("dev", dev_data)], config = {'lr': self.arg.learning_rate, 'n_epochs': self.arg.n_epochs})


	def predict(self, sentences, form):
		def prepare_data(sentences):
			j, conll_str, tsv_str = data.prepare(*sentences, form=form)
			if self.debug:
				import json
				with open("debug.json", "w", encoding="UTF8") as f:
					json.dump(j, f, ensure_ascii=False, indent="\t")
				with open("debug.conll", "w", encoding="UTF8") as f:
					for item in conll_str:
						f.write(item+"\n")
				with open("debug.tsv", "w", encoding="UTF8") as f:
					for item in tsv_str:
						f.write(item+"\n")
			dataset = D.generate_dataset_from_str(conll_str, tsv_str)
			return j, dataset, self.ranker.get_data_items(dataset)
		
		@TimeUtil.measure_time
		def _predict(j, dataset, data):
			# doc_stat = {}
			# for content in data:
			# 	print(content)
			# 	if doc_name not in doc_stat:
			# 		doc_stat[doc_name] = 0
			# 	doc_stat[doc_name] += 1
			# for k, v in doc_stat.items():
			# 	for i in j:
			# 		if j["fileName"] == k:
			# 			print(len(j["entities"], v))

			self.ranker.model._coh_ctx_vecs = []
			predictions = self.ranker.predict(data)
			e = D.eval_to_log(dataset, predictions)
			jsondump(e, "debug_prediction.json")
			return merge_item(j, e)

		if type(sentences) is str:
			sentences = [sentences]
		result = _predict(*prepare_data(sentences))
		return result