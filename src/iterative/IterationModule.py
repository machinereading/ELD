import os

from .utils import IterationArgs, EmbeddingCalculator, eval
from .. import GlobalValues as gl
from ..ds import *
from ..ec import EC
from ..el import EL
from ..ev import EV, EVAll, EVRandom, EVNone, EVGold
from ..utils import jsonload, jsondump, diriter, pickleload, pickledump
import torch
import os

class IterationModule:
	# EV --> EL Rerun model

	def __init__(self, el_model_name, ev_model_name):
		gl.logger.info("Initializing IterationModule")
		ev_redirection = {"ev_all": EVAll, "ev_random": EVRandom, "ev_none": EVNone, "ev_gold": EVGold}
		self.args = IterationArgs()
		self.args.el_model_name = el_model_name
		self.args.ev_model_name = ev_model_name
		self.el_model_name = el_model_name
		self.el_model = EL("test", el_model_name)
		self.ec_model = EC()
		# self.ec_model = EC("test", ec_model_name)
		# self.ec_model = SurfaceBasedClusterModel()
		if ev_model_name in ev_redirection:
			self.ev_model = ev_redirection[ev_model_name]()
		else:
			self.ev_model = EV("test", ev_model_name)

		# self.ev_model.validation_model.to("cuda:1")
		# self.ev_model = EVAll()
		self.ent_cal = EmbeddingCalculator(self.args)

	def run(self):
		gl.logger.info("Running IterationModule")
		gl.logger.debug("Loading corpus")

		validation_corpus = [jsonload(item) for item in diriter(self.args.validation_data_dir)]
		for item in validation_corpus:
			item["entities"] = list(filter(lambda x: x["dataType"] not in ["DATE", "TIME", "JOB"], item["entities"]))
		answer = [jsonload(item) for item in diriter(self.args.test_data_dir)]
		test_corpus = Corpus.load_corpus(answer)
		# if os.path.isfile(self.args.cluster_pickle):
		# 	gl.logger.info("Loading cluster data")
		# 	clustered = pickleload(self.args.cluster_pickle)
		# else:
		gl.logger.info("No cluster data")

		corpus = Corpus.load_corpus(validation_corpus)
		gl.logger.debug("Linking corpus")
		self.el_model(*corpus)
		jsondump(corpus.to_json(), "data/namu_iteration_link_result_with_fake.json")
		gl.logger.debug("Not linked entities: %d" % sum([len([x for x in y.entities if x.entity == "NOT_IN_CANDIDATE"]) for y in corpus]))

		gl.logger.debug("Clustering corpus")
		clustered = self.ec_model(corpus)
		del self.ec_model
		torch.cuda.empty_cache()
		# print(clustered == corpus)
		gl.logger.debug("Generated clusters: %d" % len(clustered.cluster))
		jsondump([x.to_json() for x in clustered.cluster_list], "data/namu_iteration_cluster_result_with_fake.json")
		# pickledump(corpus, self.args.cluster_pickle)

		gl.logger.debug("Validating corpus")
		if self.args.put_fake_cluster:
			gl.logger.debug("Generating fake cluster")
			print("MAX BEFORE: %d" % max([len(x) for x in clustered.cluster_list]))
			clustered = self.ev_model.dataset.generate_fake_cluster(clustered)
			print("MAX AFTER: %d" % max([len(x) for x in clustered.cluster_list]))
		validated = self.ev_model(clustered)

		del self.ev_model
		torch.cuda.empty_cache()
		jsondump([x.to_json() for x in validated.cluster_list], "data/namu_iteration_validation_result_%s_with_fake.json" % self.args.ev_model_name)

		gl.logger.debug("Generating new embedding")
		new_cluster = list(filter(lambda x: x.kb_uploadable, validated.cluster_list))
		gl.logger.debug("Validated clusters: %d" % len(new_cluster))
		if len(new_cluster) > 0:
			new_ents = []
			new_embs = self.ent_cal.calculate_cluster_embedding(new_cluster)
			for cluster in new_cluster:
				new_ents.append(cluster.target_entity)
				for token in cluster:
					self.el_model.data.surface_ent_dict.add_instance(token.surface, cluster.target_entity)

			# update entity embedding
			gl.logger.debug("Updating entity embedding")
			self.el_model.data.update_ent_embedding(new_ents, new_embs)

			# reload EL
			gl.logger.debug("Reloading ranker")
			self.el_model.model_name = self.el_model_name + "_iter"
			self.el_model.args.model_name = self.el_model_name + "_iter"
			self.el_model.reload_ranker()

		gl.logger.info("Rerunning EL Module")
		# retrain_corpus = train_corpus + validation_corpus
		# random.shuffle(retrain_corpus)
		# l = len(retrain_corpus)
		# self.el_model.train(retrain_corpus[:int(l * 0.9)], retrain_corpus[int(l * 0.9):])

		# retrain corpus ???

		# run EL again
		self.el_model(*test_corpus)
		# try:
		# 	pickledump(test_corpus, "data/iterative_result_%s_with_fake.pkl" % self.args.ev_model_name)
		# except Exception as e:
		# 	print("Error dumping pickle", e)

		jsondump(test_corpus.to_json(), "data/namu_iterative_result_%s_with_fake.json" % self.args.ev_model_name)
		eval_result, eval_detail, cluster_mapping_info = eval.evaluate(test_corpus, answer)
		for i, v in cluster_mapping_info.items():
			for item in new_cluster:
				try:
					if int(item.target_entity) == i:
						item["target_entity"] = v
				except:
					pass
		jsondump([x.to_json() for x in new_cluster], "data/namu_iteration_validation_result_%s_with_fake.json" % self.args.ev_model_name)
		jsondump(eval_result, "data/namu_iterative_score_%s_with_fake_final.json" % self.args.ev_model_name)
		jsondump(eval_detail, "data/namu_iterative_el_result_%s_with_fake.json" % self.args.ev_model_name)
