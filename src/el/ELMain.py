from .utils import data as data
from .utils import args
from .utils.postprocess import merge_item
from .mulrel_nel.ed_ranker import EDRanker
from .mulrel_nel import dataset as D
from .mulrel_nel import utils as U
from .. import GlobalValues as gl
from ..utils import TimeUtil
class EL():
	def __init__(self):
		arg = args.get_args()

		voca_emb_dir = 'data/el/embeddings/'

		word_voca, word_embeddings = U.load_voca_embs(voca_emb_dir + 'dict.word',
													  voca_emb_dir + 'word_embeddings.npy')
		snd_word_voca, snd_word_embeddings = U.load_voca_embs(voca_emb_dir + '/glove/dict.word',
														  voca_emb_dir + '/glove/word_embeddings.npy')
		entity_voca, entity_embeddings = U.load_voca_embs(voca_emb_dir + 'dict.entity',
														  voca_emb_dir + 'entity_embeddings.npy')
		
		time = TimeUtil.time_millis()
		self.ranker = EDRanker(config={'hid_dims': arg.hid_dims,
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
		  'args': arg}
		)
		TimeUtil.add_time_elem("EL_init",TimeUtil.time_millis() - time)

	def train(self, sentences):
		pass

	def eval(self, sentences):
		pass

	def __call__(self, sentences):
		@TimeUtil.measure_time
		def prepare_data(sentences):
			j, conll_str, tsv_str = data.prepare(*sentences)
			dataset = D.generate_dataset_from_str(conll_str, tsv_str)
			return j, dataset, self.ranker.get_data_items(dataset, predict=True)
		
		@TimeUtil.measure_time
		def predict(j, dataset, data):
			self.ranker.model._coh_ctx_vecs = []
			predictions = self.ranker.predict(data)
			e = D.eval_to_log(dataset, predictions)
			return merge_item(j, e)

		if type(sentences) is str:
			sentences = [sentences]
		result = predict(*prepare_data(sentences))
		return result