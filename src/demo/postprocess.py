from ..ds import Vocabulary
from .. import GlobalValues as gl
gl.logger.info("Loading types...")
types = {}
with open("kb_ref/tsv_instance_types_ko.ttl", encoding="UTF8") as f1:
	for line in f1.readlines():
		s, p, o, _ = line.strip().split("\t")
		s = s.strip("<>").replace("http://ko.dbpedia.org/resource/", "")
		if s not in types:
			types[s] = set([])
		types[s].add(o.strip("<>"))
with open("kb_ref/tsv_instance_types_transitive_ko.ttl", encoding="UTF8") as f2:
	for line in f2.readlines():
		s, p, o, _ = line.strip().split("\t")
		s = s.strip("<>").replace("http://ko.dbpedia.org/resource/", "")
		if s not in types:
			types[s] = set([])
		types[s].add(o.strip("<>"))
with open("kb_ref/wikidata_types.nt", encoding="UTF8") as f3:
	for line in f3.readlines():
		s, p, o = line.strip().split("\t")
		o = o.split(" ")[0]
		s = s.strip("<>").replace("http://ko.dbpedia.org/resource/", "")
		if s not in types:
			types[s] = set([])
		types[s].add(o.strip("<>"))

uris = {}

with open("kb_ref/tsv_interlanguage_links_ko.ttl", encoding="UTF8") as f:
	for line in f.readlines():
		s, p, o, _ = line.strip().split("\t")
		s = s.strip("<>").replace("http://ko.dbpedia.org/resource/", "")
		if "http://dbpedia.org" in o:
			uris[s] = o.strip("<>").replace("http://dbpedia.org/resource/", "")

print(len(types), len(uris))

def postprocess(entity: Vocabulary):
	print(entity.entity, entity.entity in types)
	entity.type = types[entity.entity] if entity.entity in types else []
	print(entity.type)
	entity.en_entity = uris[entity.entity] if entity.entity in uris else ""
	return entity
