# import endpoint.
# stores only global values
# set attributes by calling setattr in main module


boolmap = {"True": True, "False": False}
corpus_home = "corpus/"


def one_hot(i, total):
	i = int(i)
	result = [0 for _ in range(total)]
	result[i] = 1
	return result