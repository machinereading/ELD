from . import model_dict


class EV():
	def __init__(self, model_name):
		self.model = model_dict[model_name]


	def train(self):
		pass

	def validate(self, entity_set):
		pass