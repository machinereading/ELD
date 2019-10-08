from sklearn.metrics import f1_score

from ..utils import ELDArgs
from ...utils import readfile, pickleload
from ...ds import Corpus, CandDict
from . import DataModule
class Evaluator:
	def __init__(self, args: ELDArgs, data: DataModule):
		self.ent_list = data.ent_list
		self.redirects = pickleload(args.redirects_path)
		self.surface_ent_dict = CandDict(self.ent_list, pickleload(args.entity_dict_path), self.redirects)
		self.entity_embedding = data.entity_embedding
		self.out_kb_entity_embedding = data.out_kb_entity_embedding

	def evaluate(self, in_kb_pred, pred_embedding, in_kb_label, index):
		# in-kb items
		in_tp = 0
		in_gold = 0
		in_pred = 0

		# nil items (not yet disambiguated)
		nil_tp = 0
		nil_gold = 0
		nil_pred = 0

		# out-kb items
		out_tp = 0
		out_gold = 0
		out_pred = 0

		# items that have no surface information
		no_surface_tp = 0
		no_surface_gold = 0
		no_surface_pred = 0

		# give entity uri to each cluster
		pe_dark_id_to_ge_entity_map = {}
		for pe in pred.entity_iter():
			pass

		for ge, pe in zip(gold.entity_iter(), pred.entity_iter()):
			# in-kb items
			correct = ge.entity == pe.entity
			predicted = pe.entity not in ["NOT_AN_ENTITY", "NOT_IN_CANDIDATE"]
			if ge.entity_in_kb:
				in_gold += 1
				if predicted:
					in_pred += 1
					if correct:
						in_tp += 1
			# out-kb items
			else:
				out_gold += 1
				if predicted:
					out_pred += 1
					if correct:
						out_tp += 1
			if ge.surface not in self.surface_ent_dict:
				no_surface_gold += 1
				if predicted:
					no_surface_pred += 1
					if correct:
						no_surface_tp += 1

		return (in_tp, in_pred, in_gold), (out_tp, out_pred, out_gold), (no_surface_tp, no_surface_pred, no_surface_gold)
