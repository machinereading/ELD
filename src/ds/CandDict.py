import numpy as np

class CandDict:
	def __init__(self, kb, init_dict, redirect_dict):
		self._kb = kb
		self._dict = init_dict
		self._redirects = redirect_dict
		self._calc_dict = {}  # lazy property
		self.generate_calc_dict()

	def add_instance(self, surface, entity):
		if surface not in self._dict:
			self._dict[surface] = {}
		if entity not in self._dict[surface]:
			self._dict[surface][entity] = 0
		self._dict[surface][entity] += 1
		self.generate_calc_dict(surface)

	def generate_calc_dict(self, update_target=None):
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
				ent_key = ent.replace(" ", "_").split("_(")[0]
				try:
					s = sum(self._dict[ent_key])
					if ent in self._dict[ent_key]: pass
					else:
						self._calc_dict[ent_key][ent] = 1
				except:
					self._dict[ent_key] = {ent: 1}
			for m, e in self._dict.items():
				# e = {k: v for k, v in e.items() if k in self._kb} # filter only in-kb items
				x = list(e.values())
				values = np.around(x / np.sum(x), 4)
				self._calc_dict[m] = {}
				for i, (key, value) in enumerate(e.items()):
					self._calc_dict[m][key] = (values[i], idx)
					idx += 1

	def __getitem__(self, surface):
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
		return len(self[surface]) == 0
