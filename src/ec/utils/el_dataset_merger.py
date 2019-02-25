def generate_data_from_el_result(data):
	"""
	json -> list of vocabulary
	"""
	for entity in data["entities"]:
		surface = entity["text"]
		entity = entity["entity"]
		start = entity["start"]
		end = entity["end"]
		type = entity["ne_type"]
		
		