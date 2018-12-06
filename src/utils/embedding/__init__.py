
from .PretrainedEmbedding import PretrainedEmbedding
from .JamoEmbedding import JamoEmbedding
from .Glove import Glove
def load_embedding(target):
	return {"PretrainedEmbedding": PretrainedEmbedding, "JamoEmbedding": JamoEmbedding, "Glove": Glove}[target]