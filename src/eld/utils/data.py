import numpy as np
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .Dataset import ELDDataset
from .args import ELDArgs
from src.utils.datafunc import text_to_etri, etri_to_ne_dict
from ..modules.InKBLinker import MulRel, PEM, Dist, InKBLinker
from ..modules.TypePred import TypeGiver
from ... import GlobalValues as gl
from ...ds import *
from ...utils import *

class DataModule:
	def __init__(self, mode: str, args: ELDArgs):
		# index 0: not in dictionary
		# initialize embedding
		gl.logger.info("Initializing ELD data module")
		self.mode = mode
		self.args = args
		self.device = args.device
		self.ce_flag = args.use_character_embedding
		self.we_flag = args.use_word_context_embedding or args.use_word_embedding
		self.ee_flag = args.use_entity_context_embedding
		self.re_flag = args.use_relation_embedding
		self.te_flag = args.use_type_embedding
		self.ent_list = [x for x in readfile(args.ent_list_path)]
		self.redirects = pickleload(args.redirects_path)
		self.surface_ent_dict = CandDict(self.ent_list, pickleload(args.entity_dict_path), self.redirects)

		self.use_explicit_kb_classifier = args.use_explicit_kb_classifier
		self.modify_entity_embedding = args.modify_entity_embedding
		self.modify_entity_embedding_weight = args.modify_entity_embedding_weight
		self.use_cache_kb = args.use_cache_kb
		self.use_heuristic = args.use_heuristic
		self.type_predict = args.type_prediction
		self.cand_only = args.cand_only
		self.use_cand = True

		self.calc_threshold = lambda x: x * 0.05

		# load corpus and se
		if self.type_predict:
			self.typegiver = TypeGiver(args)
			if mode == "typeeval":
				return
		self.use_kb_relation_info = args.use_kb_relation_info
		if self.use_kb_relation_info:
			self.kg = Graph()
			for line in readfile(args.kb_relation_file):
				s, p, o = line.split("\t")
				self.kg.add_edge(s, p, o)
		self.e2i = {w: i + 1 for i, w in enumerate(readfile(args.entity_file))}
		self.e2i["NOT_IN_CANDIDATE"] = 0
		self.new_entity_idx = {}
		self.i2e = {v: k for k, v in self.e2i.items()}
		ee = np.load(args.entity_embedding_file)
		self.entity_embedding = torch.tensor(np.stack([np.zeros(ee.shape[-1]), *ee])).float()


		assert len(self.e2i) == self.entity_embedding.size(0)
		# print(len(self.e2i), len(self.i2e), self.entity_embedding.shape)
		if self.use_cache_kb:
			self.cache_entity_embedding = {i: torch.zeros([0, ee.shape[-1]], dtype=torch.float) for i in range(1, 20)}
			self.cache_entity_surface_dict = {i: [] for i in range(1, 20)}
			self.cache_entity_embedding_items = {i: [] for i in range(1, 20)}
			self.pred_entity_embedding = torch.zeros([0, ee.shape[-1]], dtype=torch.float)
			self.pred_entity_surface_dict = []
			self.pred_i2e = {}
			if os.path.isfile("cache_kb_entity_names.txt"):
				self.pred_i2e = {i: k for i, k in enumerate(readfile("cache_kb_entity_names.txt"))}
				self.pred_entity_surface_dict = jsonload("cache_kb_ent_surface_dict.json")
				self.pred_entity_embedding = torch.tensor(np.load("cache_kb_entity_embedding.npy"))
		self.original_entity_embedding = self.entity_embedding.clone()
		self.original_surface_ent_dict = CandDict(self.ent_list, pickleload(args.entity_dict_path), self.redirects)
		self.original_e2i = {k: v for k, v in self.e2i.items()}
		self.ee_dim = args.e_emb_dim = ee.shape[-1]

		self.ce_dim = self.we_dim = self.re_dim = self.te_dim = 1
		# print(len([x for x in readfile(args.train_filter)]))
		if self.ce_flag:
			self.c2i = {w: i + 1 for i, w in enumerate(readfile(args.character_file))}
			ce = np.load(args.character_embedding_file)
			self.character_embedding = torch.nn.Embedding.from_pretrained(torch.tensor(np.stack([np.zeros(ce.shape[-1]), *ce])).float())
			self.ce_dim = args.c_emb_dim = ce.shape[-1]
		if self.we_flag:
			self.w2i = {w: i + 1 for i, w in enumerate(readfile(args.word_file))}
			we = np.load(args.word_embedding_file)
			self.word_embedding = torch.nn.Embedding.from_pretrained(torch.tensor(np.stack([np.zeros(we.shape[-1]), *we])).float())
			self.we_dim = args.w_emb_dim = we.shape[-1]
		if self.re_flag:
			# relation embedding 방법
			# 1. [relation id, 상대적 position, incoming/outgoing, score] -> BATCH * WORD * (RELATION + 3) <- 이거로
			# 2. [incoming relation score, outgoing relation score, ...] -> BATCH * (WORD * 2)
			self.r2i = {w: i for i, w in enumerate(readfile(args.relation_file))}
			self.re_dim = args.r_emb_dim = len(self.r2i) + 3
			self.relation_limit = args.relation_limit

		if self.te_flag:  # one-hot?
			self.t2i = {w: i + 1 for i, w in enumerate(readfile(args.type_file))}
			self.te_dim = args.t_emb_dim = len(self.t2i) + 1

		in_kb_linker_dict = {"mulrel": MulRel, "pem": PEM, "dist": Dist}
		self.in_kb_linker: InKBLinker = in_kb_linker_dict[args.in_kb_linker](args, self.surface_ent_dict)


		self.oe2i = {}
		self.i2oe = {}
		# self.oe2i["NOT_IN_CANDIDATE"] = 0
		# self.i2oe = {v: k for k, v in self.oe2i.items()}
		self.train_oe_embedding = torch.nn.Embedding(0, 300)
		self.err_entity = set([])
			# self.corpus = Corpus.load_corpus(args.corpus_dir, limit=1000 if args.test_mode else 0, min_token=10)

			# self.initialize_corpus_tensor(self.corpus)
			# gl.logger.info("Corpus initialized")
			# self.train_dataset = ELDDataset(mode, self.corpus, args, cand_dict=self.surface_ent_dict, filter_list=[x for x in readfile(args.train_filter)], limit=args.train_corpus_limit)
			# gl.logger.debug("Train corpus size: %d" % len(self.train_dataset))
			# self.dev_dataset = ELDDataset(mode, self.corpus, args, cand_dict=self.surface_ent_dict, filter_list=[x for x in readfile(args.dev_filter)], limit=args.dev_corpus_limit)
			# gl.logger.debug("Dev corpus size: %d" % len(self.dev_dataset))
			# self.test_dataset = ELDDataset(mode, self.corpus, args, cand_dict=self.surface_ent_dict, filter_list=[x for x in readfile(args.test_filter)], limit=args.test_corpus_limit)
			# args.jamo_limit = self.train_dataset.max_jamo_len_in_word
			# args.word_limit = self.train_dataset.max_word_len_in_entity

		# self.corpus = Corpus.load_corpus(args.corpus_dir)
		# self.initialize_vocabulary_tensor(self.corpus)
		# self.test_dataset = ELDDataset(self.corpus, args)
		self.new_entity_count = 0
		self.new_ent_threshold = args.out_kb_threshold
		self.register_threshold = args.new_ent_threshold

	@TimeUtil.measure_time
	def initialize_corpus_tensor(self, corpus: Corpus, pred=False, train=True):
		for token in corpus.eld_items:
			if token.is_entity and token.entity.startswith("namu_") and token.entity not in self.oe2i:
				self.oe2i[token.entity] = len(self.oe2i)
			if token.is_entity and not token.entity.startswith("namu_") and token.entity not in self.e2i:
				token.target = False
		if train:
			self.train_oe_embedding = torch.nn.Embedding(len(self.oe2i), self.ee_dim).to(torch.float)
		self.i2oe = {v: k for k, v in self.oe2i.items()}
		for token in tqdm(corpus.token_iter(), total=corpus.token_len, desc="Initializing Tensors"):
			self.initialize_token_tensor(token, pred, train)
		if train:
			self.give_candidates(corpus)



	def initialize_token_tensor(self, token: Vocabulary, pred_mode=False, is_train_data=True):
		if token.is_entity:
			token.entity_embedding = self.entity_embedding[self.e2i[token.entity] if token.entity in self.e2i else 0]
			if self.use_kb_relation_info:
				node = self.kg[token.entity]
				token.degree = node.degree if node is not None else 0
		if self.ce_flag:
			token.char_embedding = self.character_embedding(torch.tensor([self.c2i[x] if x in self.c2i else 0 for x in token.jamo]))
		if self.we_flag:
			words = KoreanUtil.tokenize(token.surface)
			if len(words) == 0: words = [token.surface]
			token.word_embedding = self.word_embedding(torch.tensor([self.w2i[x] if x in self.w2i else 0 for x in words]))
		if not pred_mode and not token.target: return token  # train mode에서는 target이 아닌 것의 relation과 type 정보는 필요없음
		if self.re_flag:
			relations = []
			for rel in token.relation:
				x = one_hot(self.r2i[rel.relation], len(self.r2i))
				x.append(rel.relative_index)
				x.append(rel.score)
				x.append(1 if rel.outgoing else -1)
				relations.append(x[:])
			if len(relations) > 0:
				relations = sorted(relations, key=lambda x: -x[-2])[:self.relation_limit]
				token.relation_embedding = torch.tensor(np.stack(relations), dtype=torch.float)
			else:
				token.relation_embedding = torch.zeros(1, self.re_dim, dtype=torch.float)
		if self.te_flag:
			ne_type = token.ne_type[:2].upper() if token.ne_type is not None else None
			token.type_embedding = torch.tensor(one_hot(self.t2i[ne_type] if ne_type in self.t2i else 0, self.te_dim))
		if not pred_mode and token.target:  # train mode 전용 label setting
			if token.entity in self.e2i:
				if is_train_data: token.entity_label_embedding = self.entity_embedding[self.e2i[token.entity]]
				token.entity_label_idx = self.e2i[token.entity]
			elif token.entity in self.oe2i:
				if is_train_data: token.entity_label_embedding = self.train_oe_embedding(torch.tensor(self.oe2i[token.entity])).clone().detach()
				token.entity_label_idx = self.oe2i[token.entity] + len(self.e2i)
			else:
				if token.entity not in self.err_entity:
					# gl.logger.debug(token.entity + " is not in entity embedding")
					self.err_entity.add(token.entity)
				token.entity_label_embedding = torch.zeros(self.ee_dim, dtype=torch.float)
				token.entity_label_idx = -1
				if is_train_data:
					token.target = False
		return token

	def update_new_entity_embedding(self, dataset: Corpus, new_entity_flag, gold_idx, pred_emb, epoch):
		"""
		Train 과정에서 label별로 entity embedding을 업데이트 하는 함수, prediction에서는 안쓰임
		candidate도 여기서 update하는게 좋을듯
		"""
		assert self.mode == "train"

		emb_map = {}
		l = len(self.e2i)
		pred_emb = pred_emb.clone().detach()
		for flag, i, e in zip(new_entity_flag, gold_idx, pred_emb):
			i = i.item()
			if i not in emb_map:
				emb_map[i] = []
			emb_map[i].append(e)
		for k, v in emb_map.items():
			if k >= l:
				idx = k - len(self.e2i)
				update_target = self.train_oe_embedding
				result = (sum(v) / len(v)).to(self.device)

				target = self.train_oe_embedding[idx].to(self.device)
				if sum(target) == 0:
					update_target[idx] = result
				else:
					update_target[idx] = (result * (0.5 - epoch * 0.001) + target * (0.5 + epoch * 0.001)).clone().detach()  # stabilize
		for token in dataset.eld_items:
			if token.is_new_entity:
				token.entity_label_embedding = self.train_oe_embedding[token.entity_label_idx - len(self.e2i)].clone().detach()
		self.give_candidates(dataset)

	def give_candidates(self, dataset: Corpus):
		gl.logger.info("Generating candidates")
		oe_len = len(self.oe2i)
		ie_len = len(self.original_e2i)
		rands = np.random.randint(0, 10, len(dataset.eld_items))
		neg_sample_size = 10
		for token, r in zip(dataset.eld_items, rands):
			token.answer_in_candidate = r
			if token.is_new_entity:
				samples = list(range(oe_len))
				samples.pop(token.entity_label_idx - ie_len)
				nsamples = torch.tensor(np.random.choice(samples, neg_sample_size))
				emb_samples = self.train_oe_embedding(nsamples)
				emb_samples[r] = token.entity_label_embedding
			else:
				samples = list(range(ie_len))
				samples.pop(token.entity_label_idx)
				nsamples = np.random.choice(samples, neg_sample_size)
				emb_samples = torch.stack([self.entity_embedding[x] for x in nsamples])
				emb_samples[r] = token.entity_label_embedding
			token.candidiate_entity_emb = emb_samples

	def reset_new_entity(self):
		if self.use_cache_kb:
			self.cache_entity_embedding = {i: torch.zeros([0, self.ee_dim], dtype=torch.float) for i in range(1, 20)}
			self.cache_entity_surface_dict = {i: [] for i in range(1, 20)}
			self.cache_entity_embedding_items = {i: [] for i in range(1, 20)}
		else: # Threshold별로 저장하는 대신 매번 predict할때마다 reset하게 바꿈.
			self.entity_embedding = self.original_entity_embedding
			self.surface_ent_dict = self.original_surface_ent_dict
			self.e2i = self.original_e2i

	@TimeUtil.measure_time
	def predict_entity(self, target_voca_list, new_ent_pred=None, pred_embedding=None, *, output_as_idx=True, mark_nil=False, no_in_kb_link=False):
		# batchwise prediction, with entity registeration
		disable_embedding = pred_embedding is None

		if new_ent_pred is None:
			new_ent_pred = torch.zeros(len(target_voca_list), dtype=torch.uint8)

		if pred_embedding is None:
			pred_embedding = torch.zeros(len(target_voca_list), self.ee_dim, dtype=torch.float)
		result = []
		ent_result = []
		new_ent_scores = []
		in_kb_idx_queue = []
		in_kb_voca_queue = []
		sims = []
		for idx, (i, e, v) in enumerate(zip(new_ent_pred, pred_embedding, target_voca_list)):
			if type(i) is torch.Tensor:
				i = i.item()

			# get max similarity and index prediction
			if self.use_explicit_kb_classifier:  # in-KB score를 사용할 경우
				# new_ent_flag = v.is_new_entity # oracle test
				new_ent_flag = i > self.new_ent_threshold
				if new_ent_flag:  # out-kb
					if self.use_cache_kb and self.cache_entity_embedding.size(0) > 0:  # out-kb similarity
						max_sim, pred_idx = self.get_pred(e, self.cache_entity_embedding) if not disable_embedding else (0, 0)
					else:  # empty out-kb. register
						max_sim = 0
						pred_idx = -1
				else:  # in-kb
					max_sim, pred_idx = self.get_pred(e, self.entity_embedding) if not disable_embedding else (0, 0)
					in_kb_idx_queue.append(idx)
					in_kb_voca_queue.append(v)
					result.append(pred_idx)
					ent_result.append(self.i2e[pred_idx])
			else:  # in-KB score를 사용하지 않고 direct 비교
				max_sim, pred_idx = self.get_pred(e, torch.cat((self.entity_embedding, self.cache_entity_embedding))) if not disable_embedding else (0, 0)
				new_ent_flag = max_sim < self.register_threshold

			# print(new_ent_flag, max_sim, pred_idx)
			# out-kb에 대한 registration & result generation
			if new_ent_flag:  # out-kb
				if self.use_heuristic:  # heuristic
					if len(v.surface) > 3:
						out_flag = False
						for new_ent_idx, ent_dict in enumerate(self.cache_entity_surface_dict):
							if out_flag: break
							for ent_cand in ent_dict:
								if v.surface in ent_cand or ent_cand in v.surface:  # 완벽포함관계
									pred_idx = new_ent_idx
									max_sim = 1
									out_flag = True
									# print(v.surface, F.cosine_similarity(e.unsqueeze(0), self.new_entity_embedding[new_ent_idx].unsqueeze(0).to(self.device)).squeeze() - F.pairwise_distance(e.unsqueeze(0), self.new_entity_embedding[new_ent_idx].unsqueeze(0).to(self.device)).squeeze())
									break

				if max_sim < self.register_threshold:  # entity registeration
					if self.use_cache_kb:  # cache KB에 entity, entity embedding 저장 후 index 반환
						pred_idx = self.cache_entity_embedding.size(0)
						self.cache_entity_embedding = torch.cat((self.cache_entity_embedding, e.unsqueeze(0).cpu()))
						self.cache_entity_surface_dict.append({v.surface})
						assert len(self.cache_entity_surface_dict) == self.cache_entity_embedding.size(0)
					else:  # cache KB를 따로 사용하지 않고 바로 KB에 등록
						pred_idx = 0  # out-kb로 판단한 경우 e2i의 길이만큼 더하는 sequence가 밑에 있으므로 일단 0임
						l = len(self.e2i)
						ent = "_" + v.surface  # 새로운 개체명: _surface
						self.e2i[ent] = l  # e2i에 새로운 개체명 추가
						self.i2e[l]= ent
						# self.surface_ent_dict.add_instance(v.surface, ent)  # 개체명 사전에 surface 추가
						self.in_kb_linker.update_entity(v.surface, ent, e)  # in-kb linker에도 entity랑 surface, 개체 embedding 추가
				else:
					assert pred_idx >= 0
					self.cache_entity_surface_dict[pred_idx].add(v.surface)
					if self.modify_entity_embedding:
						self.cache_entity_embedding[pred_idx] = e * self.modify_entity_embedding_weight + self.cache_entity_embedding[pred_idx] * (1 - self.modify_entity_embedding_weight)
				result.append(len(self.e2i) + pred_idx)
				ent_result.append(pred_idx)
			sims.append(max_sim)
			new_ent_scores.append(i)  # new entity flag marker

		if not no_in_kb_link and len(in_kb_idx_queue) > 0:  # in-kb entity linking
			ents = self.in_kb_linker(*in_kb_voca_queue)
			assert len(ents) == len(in_kb_voca_queue) == len(in_kb_idx_queue)
			for idx, e, v in zip(in_kb_idx_queue, ents, target_voca_list):
				nil_idx = 0 if mark_nil else result[idx]
				result[idx] = self.e2i[e] if e in self.e2i and e != "NOT_IN_CANDIDATE" else nil_idx
				ent_result[idx] = e
		# print(target_voca_list[idx].surface, target_voca_list[idx].entity, e)

		assert len([x for x in result if x == -2]) == 0
		assert len(new_ent_pred) == len(result)
		assert len(new_ent_scores) == len(result)
		# print(self.new_entity_embedding.size(0))
		if not output_as_idx:
			surface_count = {}
			for f, r, v in zip(new_ent_scores, ent_result, target_voca_list):
				if f > self.new_ent_threshold:
					if r not in surface_count:
						surface_count[r] = {}
					if v.surface not in surface_count[r]:
						surface_count[r][v.surface] = 0
					surface_count[r][v.surface] += 1

			surface_count = {k: sorted(v.items(), key=lambda x: x[1], reverse=True)[0][0].replace(" ", "_") for k, v in surface_count.items()}
			for i in range(len(ent_result)):
				if new_ent_scores[i] > self.new_ent_threshold and ent_result[i] in surface_count:
					ent_result[i] = surface_count[ent_result[i]]
			return new_ent_scores, sims, ent_result
		return new_ent_scores, sims, result

	def analyze(self, eld_items, new_ent_pred, sims, idx_pred, new_ent_label, idx_label, evaluation_result):
		result = {
			"scores": {
				"KB expectation score": evaluation_result[0],
				"Total score"         : evaluation_result[1],
				"In-KB score"         : evaluation_result[2],
				"Out-KB score"        : evaluation_result[3],
				"No-surface score"    : evaluation_result[4],
				"Clustering score"    : evaluation_result[5]
			},
			"result": []
		}
		mapping_result_clustered = evaluation_result[-2]
		mapping_result_unclustered = evaluation_result[-1]

		# print(mapping_result)
		# print(len(corpus.eld_items), len(new_ent_pred), len(idx_pred), len(new_ent_label), len(idx_label))
		has_cluster = {}
		for e, pn, s, pi, ln, li in zip(eld_items, new_ent_pred, sims, idx_pred,  new_ent_label, idx_label):
			pn, pi, s, ln, li = [x.item() for x in [pn, pi, s, ln, li]]
			# print(pn, pi, ln, li)
			# print(pi in self.i2e, pi in mapping_result, pi - len(self.e2i) in self.i2oe)

			if pn > self.new_ent_threshold:
				if mapping_result_clustered[pi] > 0:
					if mapping_result_clustered[pi] not in has_cluster:
						has_cluster[mapping_result_clustered[pi]] = []
					has_cluster[mapping_result_clustered[pi]].append(pi)
					pred_clustered = self.i2oe[mapping_result_clustered[pi] - len(self.e2i)]
				else:
					c = mapping_result_clustered[pi]
					pred_clustered = "EXPECTED_IN_KB_AS_OUT_KB" if c == 0 else c

				if mapping_result_unclustered[pi] > 0:
					if mapping_result_unclustered[pi] not in has_cluster:
						has_cluster[mapping_result_unclustered[pi]] = []
					has_cluster[mapping_result_unclustered[pi]].append(pi)
					pred_unclustered = self.i2oe[mapping_result_unclustered[pi] - len(self.e2i)]
				else:
					pred_unclustered = ["EXPECTED_IN_KB_AS_OUT_KB", "CLUSTER_PREASSIGNED"][mapping_result_unclustered[pi]]

			else:
				pred_clustered = pred_unclustered = self.i2e[pi]

			result["result"].append({
				"Surface"           : e.surface,
				"Context"           : " ".join([x.surface for x in e.lctx[-5:]] + ["[%s]" % e.surface] + [x.surface for x in e.rctx[:5]]),
				"EntPredClustered"  : pred_clustered,
				"EntPredUnclustered": pred_unclustered,
				"Entity"            : e.entity,
				"NewEntPred"        : pn,
				"NewEntLabel"       : ln
			})
		return result

	def prepare(self, mode, data:Corpus, namu_only=False) -> ELDDataset:  # for prediction mode

		for entity in data.entities: entity.target = True
		self.initialize_corpus_tensor(data, pred=True)
		return ELDDataset(mode, data, self.args, cand_dict=self.surface_ent_dict, namu_only=namu_only)
		# else:
		# 	func_chain = []
		#
		# buf = []
		# for item in data:
		# 	for func in func_chain:
		# 		item = func(item)
		# 	buf.append(item)
		# corpus = Corpus.load_corpus(buf)
		#
		# for entity in corpus._entity_iter():
		# 	entity.target = len(entity.relation) > 0  # 나무위키 전용
		# self.initialize_corpus_tensor(corpus, pred=True)
		# return ELDDataset(mode, corpus, self.args, cand_dict=self.surface_ent_dict)

	def predict_entity_with_embedding_train(self, eld_items, embedding, out_kb_flags=None):
		"""
		threshold를 20단계로 나누어서 등록 및 연결 수행
		@param eld_items:
		@param embedding:
		@param out_kb_flags:
		@return:
		"""
		idx_result = {i: [] for i in range(1, 20)}
		sim_result = {i: [] for i in range(1, 20)}
		for idx in range(1, 20):
			threshold = self.calc_threshold(idx)
			if self.use_cache_kb:
				if out_kb_flags is None:
					out_kb_flags = torch.zeros(embedding.size(0), dtype=torch.uint8)
				for ent, emb, out_kb_flag in zip(eld_items, embedding, out_kb_flags):
					candidates = []
					if not out_kb_flag:
						candidates = [self.e2i[x[0]] for x in self.surface_ent_dict[ent.surface] if x[0] in self.e2i]
						if len(candidates) == 0:
							target_emb = torch.zeros(0)
						else:
							target_emb = torch.cat([self.entity_embedding[i] for i in candidates])
					else:
						if self.use_cand:
							target_emb = []
							for i, s in enumerate(self.cache_entity_surface_dict[idx]):
								if ent.surface in s:
									candidates.append(i)
									target_emb.append(self.cache_entity_embedding[idx][i])
								elif len(ent.surface) > 3:
									for stored_surface in s:
										if ent.surface in stored_surface or stored_surface in ent.surface:
											candidates.append(i)
											target_emb.append(self.cache_entity_embedding[idx][i])
											break
							if len(candidates) == 0:
								if not self.cand_only:
									target_emb = self.cache_entity_embedding[idx]
									candidates = [x for x in range(len(self.cache_entity_surface_dict[idx]))]
								else:
									target_emb = []
							else:
								target_emb = torch.cat(target_emb)
						else:
							target_emb = self.cache_entity_embedding[idx]
							candidates = [x for x in range(len(self.cache_entity_surface_dict[idx]))]

					if len(candidates) > 0:
						if target_emb.dim() == 1:
							target_emb = target_emb.view(-1, self.ee_dim)
						assert len(candidates) == target_emb.size(0)
						sim, pred_idx = self.get_pred(emb, target_emb)
						pred_idx = candidates[pred_idx] if len(candidates) > 0 else 0
					else:
						sim, pred_idx = 0, 0
					if out_kb_flag:
						if sim < threshold: # register
							pred_idx = len(self.e2i) + self.cache_entity_embedding[idx].size(0)
							self.cache_entity_embedding[idx] = torch.cat([self.cache_entity_embedding[idx], emb.cpu().clone().detach().unsqueeze(0)]).view(-1, self.ee_dim)
							self.cache_entity_surface_dict[idx].append({ent.surface})
							self.cache_entity_embedding_items[idx].append([emb.clone().detach().cpu()])
							assert self.cache_entity_embedding[idx].size(0) == len(self.cache_entity_surface_dict[idx])
							sim = -1 # 새로 등록 표시
						else: # link cache kb
							self.cache_entity_surface_dict[idx][pred_idx].add(ent.surface)
							if self.modify_entity_embedding:
								self.cache_entity_embedding_items[idx][pred_idx].append(emb.clone().detach().cpu())
								# self.cache_entity_embedding[idx][pred_idx] *= 1 - self.modify_entity_embedding_weight
								# self.cache_entity_embedding[idx][pred_idx] += emb.cpu().clone().detach() * self.modify_entity_embedding_weight
								self.cache_entity_embedding[idx][pred_idx] = sum(self.cache_entity_embedding_items[idx][pred_idx]) / len(self.cache_entity_embedding_items[idx][pred_idx])
							pred_idx += len(self.e2i)
					idx_result[idx].append(pred_idx)
					sim_result[idx].append(sim)
			else: # TODO
				self.reset_new_entity()
				for ent, emb in zip(eld_items, embedding):
					candidates = [self.e2i[x] if x in self.e2i else 0 for x in self.surface_ent_dict[ent.surface]]
					target_emb = torch.cat([self.entity_embedding[i] for i in candidates])
					sim, pred_idx = self.get_pred(emb, target_emb)
					if sim < threshold: # register
						pred_idx = self.entity_embedding.size(0)
						self.entity_embedding = torch.cat([self.entity_embedding, emb.cpu().clone().detach().unsqueeze(0)])
						self.surface_ent_dict.add_instance(ent.surface, "_"+str(pred_idx)) # temporary id
						self.e2i["_"+str(pred_idx)] = pred_idx
					elif self.modify_entity_embedding:
						self.entity_embedding[pred_idx] *= 1 - self.modify_entity_embedding_weight
						self.entity_embedding[pred_idx] += emb.cpu().clone().detach() * self.modify_entity_embedding_weight
					idx_result[idx].append(pred_idx)
					sim_result[idx].append(sim)
		return idx_result, sim_result

	def predict_entity_with_embedding_immediate(self, eld_items, embedding, out_kb_flags=None, threshold=None): # for pred
		if threshold is None:
			threshold = self.register_threshold
		result = []
		sim_result = []
		if self.use_cache_kb:
			if out_kb_flags is None:
				out_kb_flags = torch.zeros(embedding.size(0), dtype=torch.uint8)
			for ent, emb, out_kb_flag in zip(eld_items, embedding, out_kb_flags):
				candidates = []
				empty_candidate = False
				if not out_kb_flag:
					candidates = [self.e2i[x[0]] for x in self.surface_ent_dict[ent.surface] if x[0] in self.e2i]
					if len(candidates) == 0:
						empty_candidate = True
						target_emb = torch.zeros(0)
					else:
						target_emb = torch.cat([self.entity_embedding[i] for i in candidates])
				else:
					target_emb = []
					for i, s in enumerate(self.pred_entity_surface_dict):
						if ent.surface in s:
							candidates.append(i)
							target_emb.append(self.pred_entity_embedding[i])
						elif len(ent.surface) > 3:
							for stored_surface in s:
								if ent.surface in stored_surface or stored_surface in ent.surface:
									candidates.append(i)
									target_emb.append(self.pred_entity_embedding[i])
									break
					if len(candidates) == 0:
						if not self.cand_only :
							target_emb = self.pred_entity_embedding
							candidates = [x for x in range(len(self.pred_entity_surface_dict))]
					else:
						target_emb = torch.cat(target_emb)
				if len(candidates) > 0:
					target_emb = target_emb.view(-1, self.ee_dim)
					sim, pred_idx = self.get_pred(emb, target_emb)
					pred_idx = candidates[pred_idx] if not empty_candidate else 0
				else:
					sim, pred_idx = 0, 0
				if out_kb_flag:
					if sim < threshold: # register
						self.pred_entity_embedding = torch.cat([self.pred_entity_embedding, emb.cpu().clone().detach().unsqueeze(0)]).view(-1, self.ee_dim)
						self.pred_entity_surface_dict.append({ent.surface})
						ent_name = "_"+ent.surface.replace(" ", "_")
						self.pred_i2e[len(self.pred_i2e)] = ent_name
						result.append(ent_name)
						assert self.pred_entity_embedding.size(0) == len(self.pred_entity_surface_dict)
						writefile([x for x in self.pred_i2e.values()], "cache_kb_entity_names.txt")
						np.save("cache_kb_entity_embedding.npy", self.pred_entity_embedding.numpy())
					else: # link cache kb
						self.pred_entity_surface_dict[pred_idx].add(ent.surface)
						if self.modify_entity_embedding:
							self.pred_entity_embedding[pred_idx] *= 1 - self.modify_entity_embedding_weight
							self.pred_entity_embedding[pred_idx] += emb.cpu().clone().detach() * self.modify_entity_embedding_weight
							np.save("cache_kb_entity_embedding.npy", self.pred_entity_embedding.numpy())
						result.append(self.pred_i2e[pred_idx])
					jsondump(self.pred_entity_surface_dict, "cache_kb_ent_surface_dict.json")
				else:
					result.append(self.i2e[pred_idx])
				sim_result.append(sim)
		else: # TODO
			for ent, emb in zip(eld_items, embedding):
				candidates = [self.e2i[x] if x in self.e2i else 0 for x in self.surface_ent_dict[ent.surface]]
				target_emb = torch.cat([self.entity_embedding[i] for i in candidates])
				sim, idx = self.get_pred(emb, target_emb)
				if sim < self.new_ent_threshold: # register
					idx = self.entity_embedding.size(0)
					self.entity_embedding = torch.cat([self.entity_embedding, emb.cpu().clone().detach().unsqueeze(0)])
					self.surface_ent_dict.add_instance(ent.surface, "_"+str(idx)) # temporary id
					self.e2i["_"+str(idx)] = idx
				elif self.modify_entity_embedding:
					self.entity_embedding[idx] *= 1 - self.modify_entity_embedding_weight
					self.entity_embedding[idx] += emb.cpu().clone().detach() * self.modify_entity_embedding_weight
				result.append(idx)
				sim_result.append(sim)
		return result, sim_result

	def get_pred(self, tensor, emb):
		assert emb.dim() == 2
		if emb.size(0) == 0: return 0, 0
		emb = emb.to(self.device)
		expanded = tensor.expand_as(emb)
		cos_sim = F.cosine_similarity(expanded, emb)
		# dist = F.pairwise_distance(expanded, emb)
		# dist += 1 # prevent zero division
		sim = (torch.max(cos_sim) + 1) / 2
		index = torch.argmax(cos_sim, dim=-1).item()
		del expanded, cos_sim
		return sim, index
# def hierarchical_clustering(tensors: torch.Tensor):
# 	iteration = 0
# 	clustering_result = [[x] for x in range(tensors.size(0))]
# 	clustering_tensors = tensors.clone().detach()
# 	clustering_result_history = []
# 	original_state = (1, clustering_result[:], clustering_tensors.clone())
# 	while len(clustering_result) > 1:  # perform clustering until 1 remaining cluster
# 		iteration += 1
# 		iteration_max_sim = 0
# 		iteration_max_pair = None
# 		for i, tensor in enumerate(clustering_tensors[:-1]):
# 			target_tensors = clustering_tensors[i + 1:]
# 			sim = F.pairwise_distance(tensor.expand_as(target_tensors), target_tensors)
# 			max_val = torch.max(sim).item()
# 			max_idx = torch.argmax(sim).item()
# 			if max_val > iteration_max_sim:
# 				iteration_max_sim = max_val
# 				iteration_max_pair = [i, max_idx + i + 1]
#
# 		f, t = iteration_max_pair
# 		clustering_result.append(clustering_result[f] + clustering_result[t])  # add new cluster
# 		clustering_result = clustering_result[:f] + clustering_result[f + 1:t] + clustering_result[t + 1:]  # remove original cluster
# 		# print(clustering_result)
# 		t = [[tensors[x] for x in y] for y in clustering_result]  # cluster-tensor mapping
# 		clustering_tensors = torch.stack([sum(x) / len(x) for x in t])  # update tensors
#
# 		clustering_result_history.append((iteration_max_sim, clustering_result[:], clustering_tensors.clone()))
# 	# find pivot - minimum similarity
# 	target = original_state
# 	buf = original_state
# 	min_sim = 100
# 	for history in clustering_result_history:
# 		sim = history[0]
# 		if sim < min_sim:
# 			min_sim = sim
# 			target = buf
# 		buf = history
# 	_, clustering_result, clustering_tensors = target
# 	# gl.logger.debug("Pre-Clustering - Before: %d, After: %d" % (tensors.size(0), len(clustering_result)))
# 	return clustering_result, clustering_tensors
