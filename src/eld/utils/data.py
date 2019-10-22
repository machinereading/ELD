import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from .args import ELDArgs
from ..modules.InKBLinker import MulRel, PEM, Dist, InKBLinker
from ... import GlobalValues as gl
from ...ds import *
from ...utils import readfile, pickleload, TimeUtil, one_hot, KoreanUtil

class ELDDataset(Dataset):
	def __init__(self, corpus: Corpus, args: ELDArgs):
		self.corpus = corpus
		self.max_jamo_len_in_word = args.jamo_limit
		self.max_word_len_in_entity = args.word_limit
		self.max_token_len_in_sentence = max(map(len, self.corpus))
		self.device = args.device
		self.window_size = args.context_window_size

		self.ce_dim = args.c_emb_dim
		self.we_dim = args.w_emb_dim
		self.ee_dim = args.e_emb_dim
		self.re_dim = args.r_emb_dim

		self.ce_flag = args.use_character_embedding
		self.we_flag = args.use_word_embedding
		self.wce_flag = args.use_word_context_embedding
		self.ee_flag = args.use_entity_context_embedding
		self.re_flag = args.use_relation_embedding
		self.te_flag = args.use_type_embedding

		self.r_limit = args.relation_limit

	# initialize maximum

	@TimeUtil.measure_time
	def __getitem__(self, index):
		# return ce, cl, lcwe, lcwl, rcwe, rcwl, lcee, lcel, rcee, rcel, re, rl, te, tl, lab
		# entity에 대한 주변 단어? -> we와 ee는 context, re는 다른 개체와의 관계, te는 자기 type
		def pad(tensor, pad_size):
			if tensor.dim() == 1:
				tensor = tensor.unsqueeze(0)
			return F.pad(tensor, [0, 0, 0, pad_size - tensor.size()[0]])

		# return torch.cat((tensor, torch.zeros(pad_size - tensor.size()[0], emb_dim, dtype=torch.float64)))
		target = self.corpus.eld_get_item(index)

		ce, we, lwe, rwe, lee, ree, re, te, new_ent, ee_label, eidx = target.tensor
		cl = wl = lwl = rwl = lel = rel = rl = tl = 0
		if self.ce_flag:
			cl = ce.size()[0]
			ce = pad(ce, self.max_jamo_len_in_word)
		if self.we_flag:
			wl = we.size()[0]
			we = pad(we, self.max_word_len_in_entity)
		if self.wce_flag:
			if len(lwe) == 0:
				lwe = [torch.zeros(1, self.we_dim, dtype=torch.float)]
			if len(rwe) == 0:
				rwe = [torch.zeros(1, self.we_dim, dtype=torch.float)]
			lwe = torch.cat(lwe, dim=0).view(-1, self.we_dim)[-self.window_size:]
			rwe = torch.cat(rwe, dim=0).view(-1, self.we_dim)[:self.window_size]
			lwl = lwe.size()[0]
			rwl = rwe.size()[0]
			lwe = pad(lwe, self.window_size)
			rwe = pad(rwe, self.window_size)
		if self.ee_flag:
			if len(lee) == 0:
				lee = [torch.zeros(1, self.ee_dim, dtype=torch.float)]
			if len(ree) == 0:
				ree = [torch.zeros(1, self.ee_dim, dtype=torch.float)]
			lee = torch.cat(lee, dim=0).view(-1, self.ee_dim)[-self.window_size:]
			ree = torch.cat(ree, dim=0).view(-1, self.ee_dim)[:self.window_size]
			lel = lee.size()[0]
			rel = ree.size()[0]
			lee = pad(lee, self.window_size)
			ree = pad(ree, self.window_size)
		if self.re_flag:
			re = target.relation_embedding
			rl = re.size()[0]
			re = pad(re, self.r_limit)
		if self.te_flag:
			te = target.type_embedding
			tl = te.size()[0]

		return ce, cl, we, wl, lwe, lwl, rwe, rwl, lee, lel, ree, rel, re, rl, te, tl, new_ent, ee_label, eidx

	def __len__(self):
		limit = 99999
		return min(self.corpus.eld_len, limit)  # todo limit 지우기

class DataModule:
	def __init__(self, mode: str, args: ELDArgs):
		# index 0: not in dictionary
		# initialize embedding
		self.mode = mode
		self.device = args.device
		self.ce_flag = args.use_character_embedding
		self.we_flag = args.use_word_context_embedding
		self.ee_flag = args.use_entity_context_embedding
		self.re_flag = args.use_relation_embedding
		self.te_flag = args.use_type_embedding
		self.ent_list = [x for x in readfile(args.ent_list_path)]
		self.redirects = pickleload(args.redirects_path)
		self.surface_ent_dict = CandDict(self.ent_list, pickleload(args.entity_dict_path), self.redirects)
		self.use_explicit_kb_classifier = args.use_explicit_kb_classifier
		self.e2i = {w: i + 1 for i, w in enumerate(readfile(args.entity_file))}
		self.e2i["NOT_IN_CANDIDATE"] = 0
		self.new_entity_idx = {}
		self.i2e = {v: k for k, v in self.e2i.items()}
		ee = np.load(args.entity_embedding_file)
		self.entity_embedding = torch.tensor(np.stack([np.zeros(ee.shape[-1]), *ee])).float()
		self.new_entity_embedding = torch.zeros([0, ee.shape[-1]], dtype=torch.float)
		self.ee_dim = args.e_emb_dim = ee.shape[-1]

		self.ce_dim = self.we_dim = self.re_dim = self.te_dim = 1
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
			self.te_dim = args.t_emb_dim = len(self.t2i)

		in_kb_linker_dict = {"mulrel": MulRel, "pem": PEM, "dist": Dist}
		self.in_kb_linker: InKBLinker = in_kb_linker_dict[args.in_kb_linker](args)
		# load corpus and se
		if mode == "train":
			self.err_entity = set([])
			self.oe2i = {w: i + 1 for i, w in enumerate(readfile(args.out_kb_entity_file))}
			self.oe2i["NOT_IN_CANDIDATE"] = 0
			self.i2oe = {v: k for k, v in self.oe2i.items()}
			self.oe_embedding = torch.zeros(len(self.oe2i), self.ee_dim, dtype=torch.float)

			self.train_corpus = Corpus.load_corpus(args.train_corpus_dir, args.corpus_limit)
			error_count = self.initialize_vocabulary_tensor(self.train_corpus)
			gl.logger.info("Train corpus initialized, Errors: %d" % error_count)

			self.dev_corpus = Corpus.load_corpus(args.dev_corpus_dir, args.corpus_limit)
			error_count = self.initialize_vocabulary_tensor(self.dev_corpus)
			gl.logger.info("Dev corpus initialized, Errors: %d" % error_count)

			self.train_dataset = ELDDataset(self.train_corpus, args)
			gl.logger.debug("Train corpus size: %d" % len(self.train_dataset))
			self.dev_dataset = ELDDataset(self.dev_corpus, args)
			gl.logger.debug("Dev corpus size: %d" % len(self.dev_dataset))
			args.jamo_limit = self.train_dataset.max_jamo_len_in_word
			args.word_limit = self.train_dataset.max_word_len_in_entity
		else:
			self.corpus = Corpus.load_corpus(args.corpus_dir)
			self.initialize_vocabulary_tensor(self.corpus)
			self.test_dataset = ELDDataset(self.corpus, args)
		self.new_entity_count = 0
		self.new_ent_threshold = args.out_kb_threshold
		self.register_threshold = args.new_ent_threshold
		self.register_policy = args.register_policy
		self.init_check()

	def init_check(self):
		assert self.register_policy.lower() in ["fifo", "pre_cluster"]

	@TimeUtil.measure_time
	def initialize_vocabulary_tensor(self, corpus: Corpus):
		error_count = 0
		for token in tqdm(corpus.token_iter(), total=corpus.token_len, desc="Initializing Tensors"):
			if token.is_entity:
				token.entity_embedding = self.entity_embedding[self.e2i[token.entity] if token.entity in self.e2i else 0]
			if self.ce_flag:
				token.char_embedding = self.character_embedding(torch.tensor([self.c2i[x] if x in self.c2i else 0 for x in token.jamo]))
			if self.we_flag:
				words = KoreanUtil.tokenize(token.surface)
				if len(words) == 0: words = [token.surface]
				token.word_embedding = self.word_embedding(torch.tensor([self.w2i[x] if x in self.w2i else 0 for x in words]))
			if token.target:
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
					ne_type = token.ne_type[:2].upper()
					token.type_embedding = torch.tensor(one_hot(self.t2i[ne_type] if ne_type in self.t2i else 0, self.te_dim))
				if token.entity in self.e2i:
					token.entity_label_embedding = self.entity_embedding[self.e2i[token.entity]]
					token.entity_label_idx = self.e2i[token.entity]
				elif token.entity in self.oe2i:
					token.entity_label_embedding = torch.zeros(self.ee_dim, dtype=torch.float)
					token.entity_label_idx = self.oe2i[token.entity] + len(self.e2i)
				else:
					if token.entity not in self.err_entity:
						# gl.logger.debug(token.entity + " is not in entity embedding")
						self.err_entity.add(token.entity)
					error_count += 1
					token.entity_label_embedding = torch.zeros(self.ee_dim, dtype=torch.float)
					token.entity_label_idx = -1
					token.target = False
		# TODO entity embedding fix 지금은 몇개 비어있음
		# raise Exception("Entity not in both dbpedia and namu", token.entity)
		return error_count

	def update_new_entity_embedding(self, new_entity_flag, gold_idx, pred_emb):
		"""
		Train 과정에서 label별로 entity embedding을 업데이트 하는 함수, prediction에서는 안쓰임
		:param new_entity_flag:
		:param gold_idx: 
		:param pred_emb:
		:return:
		"""
		assert self.mode == "train"
		emb_map = {}
		for flag, i, e in zip(new_entity_flag, gold_idx, pred_emb):
			i = i.item()
			if flag.item():
				if i not in emb_map:
					emb_map[i] = []
				emb_map[i].append(e)
		for k, v in emb_map.items():
			self.oe_embedding[k - len(self.e2i)] = sum(v) / len(v)
		for token in self.train_corpus.eld_items:
			if token.target and token.entity in self.oe2i:
				token.entity_label_embedding = self.oe_embedding[token.entity_label_idx - len(self.e2i)].clone().detach()

	def reset_new_entity(self):
		self.new_entity_embedding = torch.zeros([0, self.ee_dim], dtype=torch.float)

	@TimeUtil.measure_time
	def predict_entity(self, new_ent_pred, pred_embedding, target_voca_list):
		# batchwise prediction, with entity registeration

		def get_pred(tensor, emb):
			expanded = tensor.expand_as(emb).to(self.device)
			cos_sim = F.cosine_similarity(expanded, emb.to(self.device))
			dist = F.pairwise_distance(expanded, emb.to(self.device))
			sim = torch.max(cos_sim - dist)
			index = torch.argmax(cos_sim - dist, dim=-1).item()
			return sim, index

		result = []
		new_ent_flags = []
		add_idx_queue = []
		add_tensor_queue = []
		in_kb_idx_queue = []
		in_kb_voca_queue = []
		for idx, (i, e, v) in enumerate(zip(new_ent_pred, pred_embedding, target_voca_list)):
			if type(i) is torch.Tensor:
				i = i.item()

			# get max similarity and index prediction
			if self.use_explicit_kb_classifier:  # in-KB score를 사용할 경우
				new_ent_flag = i > self.new_ent_threshold
				if new_ent_flag:  # out-kb
					if self.new_entity_embedding.size(0) > 0:  # out-kb similarity
						max_sim, pred_idx = get_pred(e, self.new_entity_embedding)
					else:  # empty out-kb. register
						max_sim = 0
						pred_idx = -1
				else:  # in-kb
					max_sim, pred_idx = get_pred(e, self.entity_embedding)
					in_kb_idx_queue.append(idx)
					in_kb_voca_queue.append(v)
					result.append(pred_idx)
			else:  # in-KB score를 사용하지 않고 direct 비교
				max_sim, pred_idx = get_pred(e, torch.cat((self.entity_embedding, self.new_entity_embedding)))
				new_ent_flag = max_sim < self.register_threshold

			# out-kb에 대한 registration & result generation
			if new_ent_flag:  # out-kb
				if max_sim < self.register_threshold:  # entity registeration
					if self.register_policy == "fifo":  # register immediately
						pred_idx = self.new_entity_embedding.size(0)
						self.new_entity_embedding = torch.cat((self.new_entity_embedding, pred_embedding.cpu()))
						result.append(len(self.e2i) + pred_idx)
					elif self.register_policy == "pre_cluster":  # register after clustering batch wise
						add_idx_queue.append(idx)
						add_tensor_queue.append(e)
						result.append(-2)
				else:  # out-kb index
					result.append(len(self.e2i) + pred_idx)  # new entity should have index from existing entities
			new_ent_flags.append(new_ent_flag)  # new entity flag marker

		if len(add_tensor_queue) > 0:  # perform preclustering
			len_before_register = self.new_entity_embedding.size(0)
			cluster_info, cluster_tensor = hierarchical_clustering(torch.stack(add_tensor_queue))
			for i, idx in enumerate(add_idx_queue):
				for x, c in enumerate(cluster_info):
					if i in c:
						result[idx] = len_before_register + x + len(self.e2i)
			self.new_entity_embedding = torch.cat((self.new_entity_embedding, *cluster_tensor))

		if len(in_kb_idx_queue) > 0:  # in-kb entity linking
			ents = self.in_kb_linker(*in_kb_voca_queue)
			for idx, e in zip(in_kb_idx_queue, ents):
				result[idx] = self.e2i[e] if e in self.e2i and e != "NOT_IN_CANDIDATE" else result[idx]

		assert len([x for x in result if x == -2]) == 0
		assert len(new_ent_pred) == len(result)
		assert len(new_ent_flags) == len(result)
		# print(self.new_entity_embedding.size(0))
		return new_ent_flags, result

	def analyze(self, corpus, new_ent_pred, idx_pred, new_ent_label, idx_label, evaluation_result):
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
		mapping_result = evaluation_result[-1]
		# print(mapping_result)
		# print(len(corpus.eld_items), len(new_ent_pred), len(idx_pred), len(new_ent_label), len(idx_label))
		has_cluster = {}
		for e, pn, pi, ln, li in zip(corpus.eld_items, new_ent_pred, idx_pred, new_ent_label, idx_label):
			pn, pi, ln, li = [x.item() for x in [pn, pi, ln, li]]
			# print(pn, pi, ln, li)
			# print(pi in self.i2e, pi in mapping_result, pi - len(self.e2i) in self.i2oe)

			if pn:
				if mapping_result[pi] != 0:
					if mapping_result[pi] not in has_cluster:
						has_cluster[mapping_result[pi]] = []
					has_cluster[mapping_result[pi]].append(pi)
					pred = self.i2oe[mapping_result[pi] - len(self.e2i)] + "_%d" % has_cluster[mapping_result[pi]].index(pi)
				else:
					pred = "EXPECTED_IN_KB_AS_OUT_KB"
			else:
				pred = self.i2e[pi]
			result["result"].append({
				"EntPred"    : pred,
				"Entity"     : e.entity,
				"NewEntPred" : pn,
				"NewEntLabel": ln
			})
		return result

def hierarchical_clustering(tensors: torch.Tensor):
	iteration = 0
	clustering_result = [[x] for x in range(tensors.size(0))]
	clustering_tensors = tensors.clone().detach()
	clustering_result_history = []
	original_state = (1, clustering_result[:], clustering_tensors.clone())
	while len(clustering_result) > 1:  # perform clustering until 1 remaining cluster
		iteration += 1
		iteration_max_sim = 0
		iteration_max_pair = None
		for i, tensor in enumerate(clustering_tensors[:-1]):
			target_tensors = clustering_tensors[i + 1:]
			sim = F.pairwise_distance(tensor.expand_as(target_tensors), target_tensors)
			max_val = torch.max(sim).item()
			max_idx = torch.argmax(sim).item()
			if max_val > iteration_max_sim:
				iteration_max_sim = max_val
				iteration_max_pair = [i, max_idx + i + 1]

		f, t = iteration_max_pair
		clustering_result.append(clustering_result[f] + clustering_result[t])  # add new cluster
		clustering_result = clustering_result[:f] + clustering_result[f + 1:t] + clustering_result[t + 1]  # remove original cluster
		t = [[tensors[x] for x in y] for y in clustering_result]  # cluster-tensor mapping
		clustering_tensors = torch.stack([sum(x) / len(x) for x in t])  # update tensors

		clustering_result_history.append((iteration_max_sim, clustering_result[:], clustering_tensors.clone()))
	# find pivot - minimum similarity
	target = original_state
	buf = original_state
	min_sim = 100
	for history in clustering_result_history:
		sim = history[0]
		if sim < min_sim:
			min_sim = sim
			target = buf
		buf = history
	_, clustering_result, clustering_tensors = target
	gl.logger.debug("Pre-Clustering - Before: %d, After: %d" % (tensors.size(0), len(clustering_result)))
	print(clustering_result)
	return clustering_result, clustering_tensors
