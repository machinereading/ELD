class MakeIter(object):
	def __init__(self, generator):
		self.generator = generator
	def __iter__(self):
		return self.generator