import argparse
import json
from src.el import EL
from src.eld import VectorBasedELD
from src.ds import Corpus
parser = argparse.ArgumentParser()
parser.add_argument("--el_model_name", type=str)
parser.add_argument("--eld_model_name", type=str)
parser.add_argument("--input_file", type=str)
parser.add_argument("--output_file", type=str)
parser.add_argument("--input_sentence", type=str)
args = parser.parse_args()
# load model
if args.el_model_name is not None:
	el = EL(model_name=args.el_model_name)
else:
	el = EL()
if args.eld_model_name is not None:
	eld = VectorBasedELD(model_name=args.eld_model_name)
else:
	eld = VectorBasedELD()

# load data
if args.input_file is not None:

	corpus = Corpus.load_corpus(args.input_file)
	el(*corpus)
	for token in corpus._entity_iter():
		token.entity = token.el_pred_entity
	eld(corpus, no_in_kb_link=True)
	if args.output_file is not None:
		with open(args.output_file, "w", encoding="UTF8") as f:
			json.dump(corpus.to_json(), f, ensure_ascii=False, indent="\t")
	else:
		print(corpus.to_json())
elif args.input_sentence is not None:
	corpus = eld(el(args.input_sentence, output_type=Corpus), no_in_kb_link=True).to_json()
	if args.output_file is not None:
		with open(args.output_file, "w", encoding="UTF8") as f:
			json.dump(corpus.to_json(), f, ensure_ascii=False, indent="\t")
	else:
		print(corpus)
