from ...ds import Corpus
def generate_data_from_el_result(data):
	"""
	json -> list of vocabulary
	"""
	if type(data) is dict:
		for entity in data["entities"]:
			entity = entity["entity"]
		
