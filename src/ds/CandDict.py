import numpy as np

class CandDict:
	"""
	plain text에서 개체 연결 후보를 저장하는 구조체
	surface - entity 연결 확률 p(e|m)을 구함
	"""
	def __init__(self, kb, init_dict, redirect_dict):
		"""

		@param kb: 개체 list
		@param init_dict: 표층형-개체명 연결 통계 dictionary
		@param redirect_dict: redirection 사전
		"""
		self._kb = kb #
		self._dict = init_dict
		self._redirects = redirect_dict
		self._calc_dict = {}  # lazy property
		self._generate_calc_dict()

	def add_instance(self, surface, entity):
		"""
		CandDict에 link case 추가 (surface-entity link case)
		@param surface: 추가할 surface
		@param entity: surface가 연결된 개체
		@return: None
		"""
		if surface not in self._dict:
			self._dict[surface] = {}
		if entity not in self._dict[surface]:
			self._dict[surface][entity] = 0
		self._dict[surface][entity] += 1
		self._generate_calc_dict(surface)

	def _generate_calc_dict(self, update_target=None):
		"""
		instance 추가 등으로 연결 확률을 update해야 할 때 가끔씩 호출

		@param update_target: default: None, None이 아닌 경우 해당 surface form만 업데이트
		@return: None
		"""
		self._calc_dict = {}
		idx = 0
		if update_target is not None:
			m = update_target
			e = self._dict[m]
			x = list(e.values())
			values = np.around(x / np.sum(x), 4)
			if m not in self._calc_dict:
				self._calc_dict[m] = {}
			# idxs = [x[1] for x in self._calc_dict[m].values()]

			self._calc_dict[m] = {}
			for i, (k, _) in enumerate(e.items()):
				self._calc_dict[m][k] = (values[i], 0)

		else:
			for ent in self._kb:
				try:
					s = sum(self._dict[ent])
					if ent in self._dict[ent]: pass
					else:
						self._calc_dict[ent][ent] = 1
				except:
					self._dict[ent] = {ent: 1}
				# ent_key = ent.replace(" ", "_").split("_(")[0]
				# try:
				# 	s = sum(self._dict[ent_key])
				# 	if ent in self._dict[ent_key]: pass
				# 	else:
				# 		self._calc_dict[ent_key][ent] = 1
				# except:
				# 	self._dict[ent_key] = {ent: 1}
			for m, e in self._dict.items():
				# e = {k: v for k, v in e.items() if k in self._kb} # filter only in-kb items
				x = list(e.values())
				values = np.around(x / np.sum(x), 4)
				self._calc_dict[m] = {}
				for i, (key, value) in enumerate(e.items()):
					self._calc_dict[m][key] = (values[i], idx)
					idx += 1

	def __getitem__(self, surface):
		"""
		표층형에 대해 normalize된 통계적 연결 빈도 반환.
		사용법
			>>> cd = CandDict(kb, init_dict, redirect_dict)
			>>> cd["대한민국"]
		@note surface form과 entity form이 완벽하게 일치하거나 포함 관계인 경우 일정 weight을 추가로 부여하는 중. 이 부분은 recall을 올리지만 precision을 떨어뜨릴 수 있음.
		@param surface: candidate를 받고 싶은 표층형
		@return: List of tuple. 하나의 element는 (개체명, 개체id, 확률값) 으로 이루어져 있음. id는 EL을 위해 임의 부여한 값임. list는 score 순서 내림차순으로 정렬되어 있음.
		"""
		if type(surface) is tuple:
			surface, limit = surface
		else:
			limit = 0
		candidates = self._calc_dict[surface] if surface in self._calc_dict else {}
		cand_list = []
		for cand_name, cand_score in sorted(candidates.items(), key=lambda x: -x[1][0]):
			cand_name = self._redirects[cand_name] if cand_name in self._redirects else cand_name
			if (cand_name in cand_list and cand_list[cand_name] < cand_score) or cand_name not in cand_list:
				score, cid = cand_score
				cand_list.append((cand_name, cid, score))

		if limit > 0:
			cand_list = cand_list[:limit]
		return cand_list

	def __contains__(self, surface):
		"""
		현재 candidate dictionary 안에 surface form의 연결 통계가 있는지 분석
		사용법:
			>>> "surface" in cd
		@param surface: 판별하고자 하는 surface form
		@return: bool
		"""
		return surface not in self._calc_dict

	# def __iadd__(self, other):
	# 	assert type(other) is CandDict
	# 	self._kb +=
