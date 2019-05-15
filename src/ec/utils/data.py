import collections
import json
import os
import re
from typing import List

from ..CR import conll
from ..CR.minimize import DocumentState
from ... import GlobalValues as gl
from ...ds import *
from ...utils import split_to_batch, KoreanUtil, TimeUtil

NER_dic = {'PERSON'      : 0, 'STUDY_FIELD': 5, 'THEORY': 5, 'ARTIFACTS': 5, 'ORGANIZATION': 2, 'LOCATION': 1,
           'CIVILIZATION': 5, 'DATE': 5, 'TIME': 4, 'EVENT': 3, 'ANIMAL': 5, 'PLANT': 5, 'MATERIAL': 5, 'TERM': 5,
           'JOB'         : 5, 'QUANTITY': 5, 'ETC': 5}

class DataModule:
	def __init__(self):
		self.data_idx = 0

	def convert_data(self, data, cluster_doc_size=100):
		"""
		Converts input data into model input format(json)
		:param cluster_doc_size: documents will be clusterd at maximum of cluster_doc_size
		:param data: input data, el marked
		:return: CR input format dictionary (modified form)
		"""
		if type(data) is Corpus:
			data = [self._convert_sentence_to_json(sentence) for sentence in data]
		nil_entity = self._gather_nil_entity(data)
		target_sentences = set(
				[x["parent_sentence_id"] for x in nil_entity])

		gl.logger.debug("# of nil entity: %d in %d documents" % (len(nil_entity), len(target_sentences)))
		# clustering idea: integrate every sentence and find coreference sentence
		# 일단 여기서 주변 문단과 잘 뭉쳤다고 가정
		clustered_sentences = split_to_batch([data[x] for x in target_sentences], cluster_doc_size)

		modified_documents = [self._convert_cluster_into_input(c) for c in clustered_sentences]
		conlls = []
		etris = []
		for c, e in [self._make_coref_indices(mdoc, mode="test") for mdoc in modified_documents]:
			conlls.append(c)
			etris.append(e)
		jsonlines = [self._make_jsonline(cs)[0] for cs in conlls]
		return jsonlines, etris, conlls

	def generate_training_data(self, data):
		self.data_idx = 0
		corpus = self._load_corpus(data)
		x = [self._make_coref_indices(mdoc, mode="train") for mdoc in corpus]
		print(len(x))
		conlls = []
		etris = []
		for c, e in x:
			conlls.append(c)
			etris.append(e)
		# conlls, etris = zip(*[self._make_coref_indices(mdoc, mode="train") for mdoc in modified_documents])
		jsonlines = []
		err_count = 0
		for i, item in enumerate(conlls):
			try:
				jsonlines += self._make_jsonline(item)
			except:
				gl.logger.debug("EC Data generation Error: %d" % i)
				err_count += 1
		# jsonlines = [self._make_jsonline(cs) for cs in conlls]
		print(err_count)
		print(len(jsonlines))
		return jsonlines, etris, conlls

	def _gather_nil_entity(self, corpus):
		result = []
		for i, sentence in enumerate(corpus):
			sentence["entities"] = [x for x in sentence["entities"] if
			                        x["entity"] in ["NOT_IN_CANDIDATE", "EMPTY_CANDIDATES"]]
			for e in sentence["entities"]:
				e["parent_sentence_id"] = i
				result.append(e)
		return result

	@TimeUtil.measure_time
	def _load_corpus(self, corpus_path):
		self.data_idx = 0
		doc_ID = []
		data_list = os.listdir(corpus_path)
		for data in data_list:
			ID = data.split("_")[0]
			if ID not in doc_ID:
				doc_ID.append(ID)
		exception = []
		exceptions = "_ "
		max_document_size = 10000
		for doc in doc_ID:
			modified_json = {}
			modified_json['plainText'] = ""
			modified_json['docID'] = str(doc)
			modified_json['parID'] = "1 "
			modified_json['globalSID'] = ""
			modified_json['pronouns'] = []
			modified_json['entities'] = []
			json_file = []
			name_list = []
			total_file_size = 0
			for i in range(1, 20):
				file_name = corpus_path + doc + "_" + str(i) + ".json"
				if os.path.exists(file_name):
					name_list.append(i)
					# print(file_name)
					g = open(file_name, "r", encoding="utf-8-sig")
					json_ = json.load(g)
					g.close()
					json_file.append(json_)
					total_file_size += os.path.getsize(file_name)

			if total_file_size > max_document_size:
				exception.append(doc)
				point = total_file_size / 2
				temp_file_size = 0
				middle_point = 0
				for i in name_list:
					file_name = corpus_path + doc + "_" + str(i) + ".json"
					if os.path.exists(file_name):
						temp_file_size += os.path.getsize(file_name)
						middle_point += 1
					if temp_file_size > point:
						middle_point -= 1
						middle_file_name = i
						break

			if doc in exception:
				sentence_length = 0
				entity_ID = 0
				pronoun_ID = 0
				entity_ID_list = [0] * 15
				pronoun_ID_list = [0] * 15
				for json_ in json_file[:middle_point]:
					json_parID = int(json_['parID'])
					modified_json['globalSID'] += json_['globalSID'] + ","
					modified_json['plainText'] += json_['plainText'] + " "
					# json_['entities'] = sorted(json_['entities'], key = lambda k : k['st'])
					# json_['pronouns'] = sorted(json_['pronouns'], key = lambda k : k['st'])
					#            print(modified_json['plainText'])
					for entity in json_['entities']:
						entity['id'] = entity_ID
						entity_ID += 1
						entity['st'] += sentence_length
						entity['en'] += sentence_length
						if entity['surface'][0] == "_":
							entity['surface'] = entity['surface'][1:]
							entity['st'] += 1
						if entity['surface'][-1] == "_":
							entity['surface'] = entity['surface'][:-1]
							entity['en'] -= 1
						if (entity['en'] - entity['st']) != len(entity['surface']):
							if modified_json['plainText'][entity['st']] in exceptions:
								entity['st'] += 1
							if modified_json['plainText'][entity['en'] - 1] in exceptions:
								entity['en'] -= 1
						if entity['ancestor'] != '':
							if not entity['ancestor'].startswith("-"):
								parID = int(entity['ancestor'].split("-")[0])
								index = 0
								for i in range(0, parID):
									index += entity_ID_list[i]
								entity_ID_ = int(entity['ancestor'].split("-")[1])
								entity['ancestor'] = "1-" + str(index + entity_ID_)
						modified_json['entities'].append(entity)

					for pronoun in json_['pronouns']:
						pronoun['id'] = pronoun_ID
						pronoun_ID += 1
						pronoun['st'] += sentence_length
						pronoun['en'] += sentence_length
						if pronoun['ancestor'] != '':
							if not pronoun['ancestor'].startswith("-"):
								parID = int(pronoun['ancestor'].split("-")[0])
								index = 0
								for i in range(0, parID):
									index += entity_ID_list[i]
								pronoun_ID_ = int(pronoun['ancestor'].split("-")[1])
								pronoun['ancestor'] = "1-" + str(index + pronoun_ID_)
						modified_json['pronouns'].append(pronoun)

					pronoun_ID_list[json_parID] = len(json_['pronouns'])
					entity_ID_list[json_parID] = len(json_['entities'])

					sentence_length += len(json_['plainText']) + 1
				modified_json['globalSID'] = modified_json['globalSID'][:-1]
				yield modified_json

				modified_json = {}
				modified_json['plainText'] = ""
				modified_json['docID'] = str(doc)
				modified_json['parID'] = "2"
				modified_json['globalSID'] = ""
				modified_json['pronouns'] = []
				modified_json['entities'] = []
				sentence_length = 0
				entity_ID = 0
				for json_ in json_file[middle_point:]:
					json_parID = int(json_['parID'])
					modified_json['globalSID'] += json_['globalSID'] + ","
					modified_json['plainText'] += json_['plainText'] + " "
					# json_['entities'] = sorted(json_['entities'] , key = lambda k : k['st'])
					# json_['pronouns'] = sorted(json_['pronouns'], key = lambda k : k['st'])
					for entity in json_['entities']:
						entity['id'] = entity_ID
						entity_ID += 1
						entity['st'] += sentence_length
						entity['en'] += sentence_length
						if entity['surface'][0] == "_":
							entity['surface'] = entity['surface'][1:]
							entity['st'] += 1
						if entity['surface'][-1] == "_":
							entity['surface'] = entity['surface'][:-1]
							entity['en'] -= 1
						if (entity['en'] - entity['st']) != len(entity['surface']):
							if modified_json['plainText'][entity['st']] in exceptions:
								entity['st'] += 1
							if modified_json['plainText'][entity['en'] - 1] in exceptions:
								entity['en'] -= 1
						if entity['ancestor'] != '':
							if not entity['ancestor'].startswith("-"):
								parID = int(entity['ancestor'].split("-")[0])
								index = 0
								if parID < middle_file_name:
									for i in range(0, parID):
										index += entity_ID_list[i]
									entity_ID_ = int(entity['ancestor'].split("-")[1])
									entity['ancestor'] = "1-" + str(index + entity_ID_)
								else:
									for i in range(middle_file_name, parID):
										index += entity_ID_list[i]
									entity_ID_ = int(entity['ancestor'].split("-")[1])
									entity['ancestor'] = "2-" + str(index + entity_ID_)

						modified_json['entities'].append(entity)
					entity_ID_list[json_parID] = len(json_['entities'])

					sentence_length += len(json_['plainText']) + 1
				modified_json['globalSID'] = modified_json['globalSID'][:-1]
				yield modified_json


			else:
				sentence_length = 0
				entity_ID = 0
				entity_ID_list = [0] * 15
				for json_ in json_file:
					json_parID = int(json_['parID'])
					# print(json_parID)
					modified_json['globalSID'] += json_['globalSID'] + ","
					modified_json['plainText'] += json_['plainText'] + " "
					# json_['entities'] = sorted(json_['entities'], key = lambda k:k['st'])
					# json_['pronouns'] = sorted(json_['pronouns'], key = lambda k:k['st'])
					for entity in json_['entities']:
						entity['id'] = entity_ID
						entity_ID += 1
						entity['st'] += sentence_length
						entity['en'] += sentence_length
						if entity['surface'][0] == "_":
							entity['surface'] = entity['surface'][1:]
							entity['st'] += 1
						if entity['surface'][-1] == "_":
							entity['surface'] = entity['surface'][:-1]
							entity['en'] -= 1
						if (entity['en'] - entity['st']) != len(entity['surface']):
							if modified_json['plainText'][entity['st']] in exceptions:
								entity['st'] += 1
							if modified_json['plainText'][entity['en'] - 1] in exceptions:
								entity['en'] -= 1
						if entity['ancestor'] != '':
							if not entity['ancestor'].startswith("-"):
								parID = int(entity['ancestor'].split("-")[0])
								index = 0
								for i in range(0, parID):
									index += entity_ID_list[i]
								entity_ID_ = int(entity['ancestor'].split("-")[1])
								entity['ancestor'] = "1-" + str(index + entity_ID_)
						modified_json['entities'].append(entity)

					entity_ID_list[json_parID] = len(json_['entities'])
					sentence_length += len(json_['plainText']) + 1
				modified_json['globalSID'] = modified_json['globalSID'][:-1]
				yield modified_json

	def _convert_sentence_to_json(self, sentence):
		def convert_token_to_json(token):
			return {
				"surface" : token.surface,
				"entity"  : token.entity,
				"start"   : token.char_ind,
				"end"     : token.char_ind + len(token.surface),
				"dataType": token.ne_type
			}

		return {
			"text"    : sentence.original_sentence,
			"entities": [convert_token_to_json(token) for token in sentence.entities],
			"id"      : sentence.id
		}

	def _convert_cluster_into_input(self, clustered_sentence):
		text = "\n".join([x["text"] for x in clustered_sentence])
		result = {
			"plainText"        : text,
			"ModifiedText"     : text,
			"pronouns"         : [],
			"pronoun_candidate": [],
			"entities"         : []
		}
		str_len_buf = 0
		idbuf = 0
		for sentence in clustered_sentence:
			for entity in sentence["entities"]:
				result["entities"].append({
					"id"        : idbuf,
					"ne_type"   : entity["dataType"],
					"st"        : entity["start"] + str_len_buf,
					"en"        : entity["end"] + str_len_buf,
					"surface"   : entity["surface"],
					"entityName": entity["entity"],
					"keyword"   : entity["entity"]
				})
				idbuf += 1
			str_len_buf += len(sentence["text"]) + 1
		return result

	def _set_position_character(self, nlp_result):
		new_nlp_result = []
		char_count = 0
		curr_position = 0
		for sent in nlp_result:
			new_sent = sent
			text = sent['text']
			now_morp_idx = 0
			for char in text:
				while now_morp_idx < len(new_sent['morp']) and new_sent['morp'][now_morp_idx][
					'position'] <= curr_position:
					if new_sent['morp'][now_morp_idx]['position'] == curr_position:
						# print('curr_position', curr_position, new_sent['morp'][now_morp_idx])
						new_sent['morp'][now_morp_idx]['st'] = char_count
						new_sent['morp'][now_morp_idx]['en'] = char_count + len(new_sent['morp'][now_morp_idx]['lemma'])
					# else : print(new_sent['morp'][now_morp_idx], curr_position)
					now_morp_idx += 1
				curr_position += len(str.encode(char))
				char_count += 1

			new_nlp_result.append(new_sent)

		return new_nlp_result

	@TimeUtil.measure_time
	def _make_coref_indices(self, modified_json, mode):
		etri_result = []
		for entity in modified_json['entities']:
			entity['st_modified'] = entity['st']
			entity['en_modified'] = entity['en']
		text = modified_json['ModifiedText'] if "ModifiedText" in modified_json else modified_json[
			"plainText"].replace("_", " ")
		tokenized = KoreanUtil.tokenize(text)
		pos = []
		last_ind = 0
		for token in tokenized:
			ind = text.index(token, last_ind)
			pos.append(ind)
			last_ind = ind + len(token)
		morp = []

		for t, i in zip(tokenized, pos):
			morp.append({
				"lemma": t,
				"st"   : i,
				"en"   : i + len(t),
				"type" : ""
			})
		tokenize_result = {
			"text": text,
			"morp": morp
		}
		etri_result.append(tokenize_result)
		morp_result = [tokenize_result["morp"]]
		self.data_idx += 1
		return self._get_conll_format(morp_result, modified_json['entities'], mode), etri_result

	def _get_conll_format(self, morp_file, entities, mode):
		fname = str(self.data_idx)
		string = "#begin document (" + fname + "); part 000\n"

		for sentence in morp_file:
			opener = 0
			closer = 0
			token_list = []
			token_index = 0
			for token in sentence:
				temp = [token['lemma'], token['type'], token['st'], token['en'], "-", "-", "*"]
				token_list.append(temp)

			entity_index = 0
			for entity in entities:
				entity["st_modified"] = entity["st"]
				entity["en_modified"] = entity["en"]
				entity_index += 1
				if mode == "train":
					if "coref_index" in entity:
						min_st = entity['st_modified']
						max_en = entity['en_modified']

						for token in token_list:
							if entity['st_modified'] <= token[2] < entity['en_modified'] and max_en <= \
									token[3]:
								max_en = token[3]
							if entity['st_modified'] < token[3] <= entity['en_modified'] and min_st >= \
									token[2]:
								min_st = token[2]

						pos = 0
						for token in token_list:
							if min_st == token[2] and max_en == token[3]:
								opener += 1
								closer += 1
								if token[4] == '-':
									token[4] = "(" + str(entity['coref_index']) + ")"
								else:
									token[4] = "(" + str(entity['coref_index']) + ")" + "|" + token[4]
								pos = 1
						if pos == 0:
							open_i = 0
							close_i = 0
							for token in token_list:
								if min_st == token[2] and open_i == 0:
									opener += 1
									if token[4] == '-':
										token[4] = "(" + str(entity['coref_index'])
									else:
										token[4] = "(" + str(entity['coref_index']) + "|" + token[4]
									open_i = 1
								if max_en == token[3] and close_i == 0:
									closer += 1
									if token[4] == '-':
										token[4] = str(entity['coref_index']) + ")"
									else:
										token[4] = token[4] + "|" + str(entity['coref_index']) + ")"
									close_i = 1
				min_st = entity['st_modified']
				max_en = entity['en_modified']

				for token in token_list:
					if entity['st_modified'] <= token[2] < entity['en_modified'] and max_en <= token[3]:
						max_en = token[3]
					if entity['st_modified'] < token[3] <= entity['en_modified'] and min_st >= token[2]:
						min_st = token[2]

				pos = 0
				for token in token_list:
					if min_st == token[2] and max_en == token[3]:
						opener += 1
						closer += 1
						if token[5] == '-':
							token[5] = "<" + str(entity_index) + ">"
						else:
							token[5] = "<" + str(entity_index) + ">" + "|" + token[5]
						if token[6] == '*':
							token[6] = "[" + str(NER_dic[entity['ne_type']]) + "]"
						else:
							token[6] = "[" + str(NER_dic[entity['ne_type']]) + "]" + "|" + token[6]
						pos = 1
				if pos == 0:
					open_i = 0
					close_i = 0
					for token in token_list:
						if min_st == token[2] and open_i == 0:
							opener += 1
							if token[5] == '-':
								token[5] = "<" + str(entity_index)
							else:
								token[5] = "<" + str(entity_index) + "|" + token[5]
							if token[6] == '*':
								token[6] = "[" + str(NER_dic[entity['ne_type']])
							else:
								token[6] = "[" + str(NER_dic[entity['ne_type']]) + "|" + token[6]
							open_i = 1
						if max_en == token[3] and close_i == 0:
							closer += 1
							if token[5] == '-':
								token[5] = str(entity_index) + ">"
							else:
								token[5] = token[5] + "|" + str(entity_index) + ">"
							if token[6] == '*':
								token[6] = str(NER_dic[entity['ne_type']]) + "]"
							else:
								token[6] = token[6] + "|" + str(NER_dic[entity['ne_type']]) + "]"
							close_i = 1

			# if opener != closer:
			# 	print("filename :", "")
			# 	print(opener, closer)
			# 	with open('make_conll.log', 'a', encoding='utf-8') as f:
			# 		log_string = "" + '\t' + str(opener) + '\t' + str(closer) + '\n'
			# 		f.write(log_string)

			for token in token_list:
				token[1] = "POS_DUMMY"
				string += fname + "\t"
				string += "0" + "\t"
				string += str(token_index) + "\t"
				token_index += 1
				string += token[0] + "\t"
				string += token[1] + "\t"
				string += "-" + "\t"
				if token[1].startswith("V"):
					string += token[0] + "\t"
				else:
					string += "-" + "\t"

				string += "-\t-\t-\t"
				string += token[6] + "\t" + token[5] + "\t"
				string += "NOTIME\tNOTIME\t-\t"
				string += token[4] + "\n"
			string += "\n"
		string += "#end document\n"
		return string

	@TimeUtil.measure_time
	def _make_jsonline(self, conll_str):
		document_state = DocumentState()
		labels = collections.defaultdict(set)
		stats = collections.defaultdict(int)
		result = []
		for line in conll_str.split("\n"):
			document = self._handle_line(line, document_state, "korean", labels, stats)
			if document is not None:
				result.append(document)
				document_state = DocumentState()
		return result

	# return "\n".join([json.dumps(x) for x in result])

	def _handle_line(self, line, document_state, language, labels, stats):
		def normalize_word(word):
			if word == "/." or word == "/?":
				return word[1:]
			return word

		begin_document_match = re.match(conll.BEGIN_DOCUMENT_REGEX, line)
		if begin_document_match:
			document_state.assert_empty()
			document_state.doc_key = conll.get_doc_key(begin_document_match.group(1), begin_document_match.group(2))
			# print(document_state.doc_key)
			return None
		elif line.startswith("#end document"):
			# document_state.assert_finalizable()
			finalized_state = document_state.finalize()
			stats["num_clusters"] += len(finalized_state["clusters"])
			stats["num_mentions"] += sum(len(c) for c in finalized_state["clusters"])
			labels["{}_const_labels".format(language)].update(l for _, _, l in finalized_state["constituents"])
			# labels["ner"].update(l for _, _, l in finalized_state["ner"])
			return finalized_state
		else:
			row = line.split()
			if len(row) == 0:
				stats["max_sent_len_{}".format(language)] = max(len(document_state.text),
				                                                stats["max_sent_len_{}".format(language)])
				stats["num_sents_{}".format(language)] += 1
				document_state.sentences.append(tuple(document_state.text))
				del document_state.text[:]
				document_state.speakers.append(tuple(document_state.text_speakers))
				del document_state.text_speakers[:]
				return None
			assert len(row) >= 12

			word = normalize_word(row[3])
			# POS = row[4]
			# head_POS = row[7]
			speaker = row[9]
			ner = row[10]
			# print(row)
			st_time = -1 if (row[-4] == 'NOTIME') else int(row[-4])
			en_time = -1 if (row[-3] == 'NOTIME') else int(row[-3])
			video_npy_file = row[-2]
			coref = row[-1]
			entity = row[-5]

			word_index = len(document_state.text) + sum(len(s) for s in document_state.sentences)
			document_state.text.append(word)
			document_state.text_speakers.append(speaker)
			# document_state.POS.append(pos)
			# document_state.head_POS.append(head_POS)
			if (len(document_state.start_times) == 0 or (
					not (document_state.start_times[-1] == st_time and document_state.end_times[-1] == en_time))):
				document_state.start_times.append(st_time)
				document_state.end_times.append(en_time)
				document_state.video_npy_files.append(video_npy_file)
			# print(word_index, parse)
			# handle_bit(word_index, parse, document_state.const_stack, document_state.constituents)
			# handle_bit(word_index, ner, document_state.ner_stack, document_state.ner)
			# coref_number = 0
			# entity_number = 0
			if coref != "-":
				for segment in coref.split("|"):
					if segment[0] == "(":
						if segment[-1] == ")":
							cluster_id = int(segment[1:-1])
							document_state.clusters[cluster_id].append((word_index, word_index))
						# coref_number += 1
						else:
							cluster_id = int(segment[1:])
							document_state.coref_stacks[cluster_id].append(word_index)
					else:
						cluster_id = int(segment[:-1])
						# print(segment,cluster_id)
						start = document_state.coref_stacks[cluster_id].pop()
						# coref_number += 1
						document_state.clusters[cluster_id].append((start, word_index))
			if entity != "-":
				for segment in entity.split("|"):
					if segment[0] == "<":
						if segment[-1] == ">":
							entity_id = int(segment[1:-1])
							document_state.entities.append((word_index, word_index, entity_id))
						# entity_number += 1
						else:
							entity_id = int(segment[1:])
							document_state.entity_stacks[entity_id].append(word_index)
					else:
						entity_id = int(segment[:-1])
						# print(segment,entity_id)
						# print(document_state.entity_stacks[entity_id])
						start = document_state.entity_stacks[entity_id].pop()
						# entity_number += 1
						document_state.entities.append((start, word_index, entity_id))
			if ner != "*":
				for segment in ner.split("|"):
					if segment[0] == "[":
						if segment[-1] == "]":
							ner_id = int(segment[1:-1])
							document_state.ners.append((word_index, word_index, ner_id))
						else:
							ner_id = int(segment[1:])
							document_state.ner_stacks[ner_id].append(word_index)
					else:
						ner_id = int(segment[:-1])
						# print(segment, ner_id)
						start = document_state.ner_stacks[ner_id].pop()
						document_state.ners.append((start, word_index, ner_id))

			return None

	def postprocess(self, input_jsonline, input_etri, prediction: List[dict]):
		result = []
		for jline, etri, data in zip(input_jsonline, input_etri, prediction):
			coref_indices = data['predicted_clusters']
			result_morp = []
			for sentence in etri:
				for morp in sentence['morp']:
					result_morp.append(morp)
			coref_num = 0
			for coref_index in coref_indices:
				coref_num += 1
				entity_name = ""
				for mention in coref_index:
					st = mention[0]
					en = mention[1]
					st_mention = result_morp[st]['st']
					en_mention = result_morp[en]['en']

					for entities in jline['entities']:
						if entities['st_modified'] == st_mention and \
								entities['en_modified'] == en_mention:
							entities['predicted_coref_index'] = coref_num

				if entity_name == "":
					st = coref_index[0][0]
					en = coref_index[0][1]
					st_mention = result_morp[st]['st']
					en_mention = result_morp[en]['en']

					for entities in jline['entities']:
						if entities['st_modified'] == st_mention and \
								entities['en_modified'] == en_mention:
							entity_name = entities['surface']
							entities['predicted_coref_index'] = coref_num

				for mention in coref_index:
					st = mention[0]
					en = mention[1]
					st_mention = result_morp[st]['st']
					en_mention = result_morp[en]['en']
					for entities in jline['entities']:
						if entities['st_modified'] == st_mention and \
								entities['en_modified'] == en_mention:
							entities['predicted_coref_index'] = coref_num

			# PronounExchangedText = jline['ModifiedText']
			# for entities in jline['entities']:
			# 	entities['st_exchanged'] = entities['st_modified']
			# 	entities['en_exchanged'] = entities['en_modified']
			#
			# for entities in jline['entities']:
			# 	if PronounExchangedText[entities['st_exchanged']:entities['en_exchanged']] != entities['surface']:
			# 		PronounExchangedText = PronounExchangedText[:entities['st_exchanged']] + entities[
			# 			'surface'] + PronounExchangedText[entities['en_exchanged']:]
			# jline['PronounExchangedText'] = PronounExchangedText

			result.append(jline)
		return result
