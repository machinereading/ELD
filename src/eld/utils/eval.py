from sklearn.metrics import f1_score

from src.el.utils.data import CandDict
from src.eld.utils import ELDArgs
from src.utils import readfile, pickleload
from ...ds import Corpus

class Evaluator:
	def __init__(self, args: ELDArgs):
		self.ent_list = [x for x in readfile(args.ent_list_path)]
		self.redirects = pickleload(args.redirects_path)
		self.surface_ent_dict = CandDict(self.ent_list, pickleload(args.entity_dict_path), self.redirects)

	def evaluate(self, gold: Corpus, pred: Corpus):
		# in-kb items
		in_gold = 0

		# out-kb items = dark entities
		out_gold = 0

		# items that have no surface information
		no_surface_gold = 0

		for gs, ps in zip(gold, pred):
			for gt, pt in zip(gs, ps):
				# in-kb items
				if gt.entity_in_kb:
					in_gold += 1

				# out-kb items
				else:
					out_gold += 1

				if gt.surface not in self.surface_ent_dict:
					no_surface_gold += 1

