from gensim.models import KeyedVectors, Word2Vec, FastText
from src.utils import KoreanUtil, readfile, writefile, TimeUtil
import os
def train_embedding():
	main_file = "corpus/ec_corpus.txt"
	target_file = "data/ec/ec_embedding.w2v"
	target_file2 = "data/ec/ec_embedding.ftt"
	with TimeUtil.TimeChecker("load doc"):
		doc = [KoreanUtil.stem_sentence(x) for x in readfile(main_file)]
		print(len(doc))
	with TimeUtil.TimeChecker("w2v"):
		w2v = Word2Vec(doc, size=50, sg=1, min_count=2)
		w2v.save(target_file)
	with TimeUtil.TimeChecker("ftt"):
		ftt = FastText(doc, size=50, min_count=2)
		ftt.save(target_file2)
	TimeUtil.time_analysis()

def make_word_embedding_file(corpus_dir, emb):
	target_words = set([])
	for item in os.listdir(corpus_dir):
		for line in readfile(corpus_dir+item):
			_, s, o, _, _ = line.split("\t")
			target_words.add(s)
	embs = []
	for word in target_words:
		x = [emb[w] for w in word.split() if w in emb]
		s = [0] * 50
		for e in x:
			for i in range(50):
				s[i] += e[i]
		embs.append(" ".join([word.replace(" ", "_")]+[str(x) for x in s]))
	writefile(embs, "data/ec/ftt.embed")
if __name__ == '__main__':
	# train_embedding()
	emb = KeyedVectors.load("data/ec/ec_embedding.ftt")
	make_word_embedding_file("../ref/mulrel-nel/el_result/", emb)