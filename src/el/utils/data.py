from . import datafunc
from ..mulrel_nel import utils as U
from ...utils import KoreanUtil, TimeUtil, readfile, pickleload
from ...ds import *

from tqdm import tqdm
import numpy as np

import random
import re
import string

class DataModule():
	def __init__(self, args):
		self.ent_list = [x for x in readfile(args.ent_list_path)]
		self.redirects = pickleload(args.redirects_path)
		self.surface_ent_dict = CandDict(pickleload(args.entity_dict_path), self.redirects)
		self.word_voca, self.word_embedding = U.load_voca_embs(args.word_voca_path, args.word_embedding_path)
		self.snd_word_voca, self.snd_word_embedding = U.load_voca_embs(args.snd_word_voca_path, args.snd_word_embedding_path)
		self.entity_voca, self.entity_embedding = U.load_voca_embs(args.entity_voca_path, args.entity_embedding_path)

	def update_ent_embedding(self, entity_voca, entity_embedding):
		if type(entity_embedding) is list:
			assert len(entity_voca) == len(entity_embedding)
		else:
			assert len(entity_voca) == entity_embedding.shape[0]
		if type(entity_embedding) is list:
			entity_embedding = np.array(entity_embedding)
		self.entity_voca += entity_voca
		# print(self.entity_embedding.shape, entity_embedding.shape)
		self.entity_embedding = np.concatenate([self.entity_embedding, entity_embedding], axis=0)

	def sentence_to_json(self, sentence):
		assert type(sentence) is Sentence
		entities = [x.to_json() for x in sentence.tokens if x.is_entity]
		for entity in entities:
			entity["candidates"] = self.surface_ent_dict[entity["surface"]]
			entity["start"] = entity["char_ind"]
			entity["end"] = entity["char_ind"] + len(entity["surface"])
		return {
			"text": sentence.original_sentence,
			"entities": entities,
			"fileName": str(sentence.id)
		}

	def make_json(self, ne_marked_dict, predict=False):
		cs_form = {}
		cs_form["text"] = ne_marked_dict["original_text"] if "original_text" in ne_marked_dict else ne_marked_dict["text"]
		cs_form["entities"] = []
		cs_form["fileName"] = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(7)) if "fileName" not in ne_marked_dict or ne_marked_dict["fileName"] == "" else ne_marked_dict["fileName"]
		entities = ne_marked_dict["NE"] if "NE" in ne_marked_dict else ne_marked_dict["entities"]
		for item in entities:
			skip_flag = False
			if "type" in item:
				for prefix in ["QT", "DT"]: # HARD-CODED: ONLY WORKS FOR ETRI TYPES
					if item["type"].startswith(prefix): skip_flag = True
				if item["type"] in ["CV_RELATION", "TM_DIRECTION"] or skip_flag: continue
			surface = item["text"] if "text" in item else item["surface"]
			if "keyword" in item or "entity" in item:
				keyword = "NOT_IN_CANDIDATE" if predict else (item["keyword"] if "keyword" in item else item["entity"])
			else:
				keyword = "NOT_IN_CANDIDATE"
			# if keyword == "NOT_AN_ENTITY":
			# 	keyword = "NOT_IN_CANDIDATE"
			if "dark_entity" in item:
				keyword = "DARK_ENTITY"
			start = item["char_start"] if "char_start" in item else item["start"]
			end = item["char_end"] if "char_end" in item else item["end"]
			if not any(list(map(KoreanUtil.is_korean_character, surface))): continue # non-korean filtering
			
			cs_form["entities"].append({
				"surface": surface,
				"candidates": self.surface_ent_dict[surface],
				"answer": keyword,
				"start": start,
				"end": end,
				"ne_type": item["type"] if "type" in item else ""
				})
		return cs_form



	def generate_input(self, sentence, predict=False):
		preprocess = {str: datafunc.mark_ne, Sentence: self.sentence_to_json, dict: lambda x: x}
		sentence = self.make_json(preprocess[type(sentence)](sentence), predict=predict)
		
		# at this point, sentence should be in Crowdsourcing form
		result = []
		links = []
		print_flag = False
		# sentence["entities"] = list(filter(lambda entity: (redirects[entity["keyword"]] if entity["keyword"] in redirects else entity["keyword"]) in ent_form, sentence["entities"]))
		for entity in sentence["entities"]:
			
			redirected_entity = self.redirects[entity["answer"]] if entity["answer"] in self.redirects else entity["answer"]
			
			# if redirected_entity not in ent_form:
			# 	redirected_entity = "NOT_IN_ENTITY_LIST"
			# if "dark_entity" in entity:
			# 	print(entity["surface"])
			# 	redirected_entity = "DARK_ENTITY"
			entity["answer"] = redirected_entity
			# if redirected_entity not in ent_form and redirected_entity not in ["NOT_IN_CANDIDATE", "NOT_AN_ENTITY", "EMPTY_CANDIDATES"]:
			# 	continue
			links.append((entity["surface"], redirected_entity, entity["start"], entity["end"], tuple(entity["candidates"])))
			
		filter_entity = set([])
		for i1 in links:
			if i1 in filter_entity: continue
			for i2 in links:
				if i1 == i2: continue
				if i1[2] <= i2[2] < i1[3] or i1[2] < i2[3] <= i1[3]:
					# overlaps
					shorter = i1 if i1[3] - i1[2] <= i2[3] - i2[2] else i2
					filter_entity.add(shorter)
		links = list(filter(lambda x: x not in filter_entity, links))
		links = sorted(links, key=lambda x: x[2])
		sent = sentence["text"]
		for char in "   ":
			sent.replace(char, " ")
		
		sent = datafunc.RE_EMOJI.sub(r'', sent)
		# morphs = datafunc.okt.morphs(sent)
		morphs = KoreanUtil.tokenize(sent)
		inds = []
		last_char_ind = 0
		conlls = []
		tsvs = []
		fname = sentence["fileName"]
		conlls.append("-DOCSTART- (%s" % fname)
		for item in morphs:
			ind = sent.find(item, last_char_ind) 
			inds.append(ind)
			last_char_ind = ind+len(item)
		assert(len(morphs) == len(inds))
		last_link = None
		added = []
		for morph, pos in zip(morphs, inds):
			for m, link in datafunc.morph_split((morph, pos), links):
				if link is None: # if train mode, skip if candidate set is empty
					conlls.append(m)
					last_link = None
					continue
				last_label = conlls[-1][1] if len(conlls) > 0 and type(conlls[-1]) is not str else "O"
				bi = "I" if last_label != "O" and last_link is not None and link == last_link else "B"
				ne, en, sp, ep, cand = link
				last_link = link
				assert m in ne, "%s, %s, %s" % (sentence, m, ne)
				conlls.append([m, bi, ne, en, "%s%s" % (datafunc.dbpedia_prefix, en), "000", "000"])
				if bi == "B":
					added.append(link)
		# not_added = list(filter(lambda x: x not in added, links))
		# if len(not_added) > 0:
		# 	print(not_added)

		for ne, en, sp, ep, cand in added:
			f = [fname, fname, ne, datafunc.get_context_words(sent, sp, -1), datafunc.get_context_words(sent, ep-1, 1), "CANDIDATES"]
			cand_list = []
			gold_ind = -1
			ind = 0
			for cand_name, cand_id, cand_score in sorted(cand, key=lambda x: -x[-1]):
				cand_list.append((self.redirects[cand_name] if cand_name in self.redirects else cand_name, cand_id, cand_score))
			if en in ["NOT_IN_CANDIDATE", "NOT_AN_ENTITY"]: en = "#UNK#"
			for cand_name, cand_id, cand_score in cand_list:
				# print(cand_score)
				f.append(",".join([str(cand_id), str(cand_score), cand_name])) # order: ID SCORE ENTITY
				if cand_name == en:
					gold_ind = ind
					gold_sent = f[-1]
				ind += 1
			if len(cand_list) == 0:
				f.append("EMPTYCAND")
			f.append("GT:")
			f.append("%d,%s" %(gold_ind, gold_sent) if gold_ind != -1 else "-1")
			tsvs.append("\t".join(f))


		conlls = list(map(lambda x: x if type(x) is str else "\t".join(x), conlls))
		if conlls[-1] in ["", "\n"]:
			conlls = conlls[:-1]
		return sentence, conlls, tsvs

	def prepare(self, *sentences, predict=False, filter_rate=0.0):
		conlls = []
		tsvs = []
		cw_form = []
		for sentence in tqdm(sentences, desc="Formatting input"):
			s = random.random()
			
			if filter_rate > s: continue
			
			try:
				j, c, t = self.generate_input(sentence, predict)
				# conll = change_to_conll(sentence)
				# tsv = change_to_tsv(sentence)
				cw_form.append(j)
				conlls += c
				conlls += [""]
				tsvs += t
			except Exception as e:
				import traceback
				traceback.print_exc()
		return cw_form, conlls, tsvs



def basic_prob(x):
	return 

class CandDict():
	def __init__(self, init_dict, redirect_dict):
		self._dict = init_dict
		self._redirects = redirect_dict
		self._calc_dict = {} # lazy property
		self._update_flag = False
	
	def add_instance(self, surface, entity):
		if surface not in self._dict:
			self._dict[surface] = {}
		if entity not in self._dict[surface]:
			self._dict[surface][entity] = 0
		self._dict[surface][entity] += 1
		self._update_flag = False

	def generate_calc_dict(self):
		self._calc_dict = {}
		idx = 0
		for m, e in self._dict.items():
			x = list(e.values())
			values = np.around(x / np.sum(x), 4)
			self._calc_dict[m] = {}
			for i, (key, value) in enumerate(e.items()):
				self._calc_dict[m][key] = (values[i], idx)
				idx += 1
		self._update_flag = True

	def __getitem__(self, item):
		if not self._update_flag:
			self.generate_calc_dict()
		candidates = self._calc_dict[item] if item in self._calc_dict else {}
		cand_list = []
		for cand_name, cand_score in sorted(candidates.items(), key=lambda x: -x[1][0]):
			cand_name = self._redirects[cand_name] if cand_name in self._redirects else cand_name
			if (cand_name in cand_list and cand_list[cand_name] < cand_score) or cand_name not in cand_list:
				score, id = cand_score
				cand_list.append((cand_name, id, score))
		return cand_list
