import json
class RunUnit():
	def __init__(self):
		pass

	def load(self, json_file):
		j = json.load(json_file)
		for k, v in j.items():
			setattr(self, k, v)