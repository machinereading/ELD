from gensim.models.ldamulticore import LdaMulticore
from ... import GlobalValues as gl
class LDA():
	def __init__(self, clustering_size):
		self.clustering_size = clustering_size

	def train_model(self, corpus):
		"""
			corpus: list of Vocabulary object
			return: list of cluster
		"""
		wc = self.calculate_frequency(corpus)

		self.model = LdaMulticore(wc, num_topics=self.clustering_size, id2word=gl.id_entity_map, workers=6)
		self.model.save("data/ec/lda")

	def calculate_frequency(self, corpus):
		wc = {}
		result = []
		for sent in corpus:
			for vocab in sent:
				for ctx in (vocab.rctx + vocab.lctx):
					if ctx.entity is not None and ctx.entity in gl.entity_id_map:
						if gl.entity_id_map[ctx.entity] not in wc:
							wc[gl.entity_id_map[ctx.entity]] = 0
						wc[gl.entity_id_map[ctx.entity]] += 1
				result.append([(int(k), v) for k, v in wc.items()])
		return result

	def get_topic(self, sent):
		for vocab_freq in self.calculate_frequency([sent]):
			yield self.model.get_document_topics(vocab_freq)

	@classmethod
	def load_from_model(cls, model_path):
		lda = LDA()
		lda.model = LdaMulticore(num_topics=self.clustering_size, id2word=gl.id_entity_map, workers=6)

if __name__ == '__main__':
	from ...utils import jsonload
	from ...utils.datamodule.CS2Sent import crowdsourcing2sent
	import os
	# corpus generate
	corpus = []
	train_d = "corpus/crowdsourcing_formatted/"
	test_d = "corpus/el_golden_postprocessed_marked/"
	for item in os.listdir(train_d):
		corpus.append(crowdsourcing2sent(jsonload(train_d+item)))
	lda = LDA(10)
	lda.train_model(corpus)
	t_corpus = []
	for item in os.listdir(test_d):
		t_corpus.append(crowdsourcing2sent(jsonload(test_d+item)))

	for sent in t_corpus:
		for item in lda.get_topic(sent):
			print(item)
