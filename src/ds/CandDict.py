import numpy as np

class CandDict:
	def __init__(self, kb, init_dict, redirect_dict):
		self._kb = kb
		self._dict = init_dict
		self._redirects = redirect_dict
		self._calc_dict = {}  # lazy property
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
		for ent in self._kb:
			try:
				s = sum(self._dict[ent])
				if ent in self._dict[ent]:
					self._calc_dict[ent][ent] += s // 5
				else:
					self._calc_dict[ent][ent] = max(s // 5, 1)
			except:
				self._dict[ent] = {ent: 1}
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
				score, cid = cand_score
				cand_list.append((cand_name, cid, score))
		return cand_list

	def __contains__(self, item):
		return len(self[item]) == 0
