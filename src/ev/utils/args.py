from ...utils import jsonload, jsondump
class EV_Args():
	def __init__(self):
		self.fake_er_rate = 0.1
		self.fake_el_rate = 0.1
		self.fake_ec_rate = 0.1
		self.data_path = "corpus/el_wiki/wiki_cwform.json"

	@classmethod
	def from_json(cls, json_file):
		args = EV_Args()
		if type(json_file) is str:
			json_file = jsonload(json_file)
		for attr, value in json_file:
			setattr(args, attr, value)
		return args

	def to_json(self):
		return self.__dict__
