import networkx as nx
import community

from src.utils import jsonload, jsondump

def cluster(corpus):
	result = []

	for item in corpus:
		add_flag = True
		new_node = Node(item["surface"], item["neighbors"], item["fileName"])
		for node in result:
			if new_node.source == node.source and new_node.surface == node.surface:
			# if new_node.surface == node.surface:
				node.neighbor = node.neighbor.union(new_node.neighbor)
				add_flag = False
			dup = len(node.neighbor_entities.intersection(new_node.neighbor_entities))
			if dup > 0 and add_flag:
				# print(dup)
				node.neighbor.add((new_node, dup))
				new_node.neighbor.add((node, dup))
		if add_flag:
			result.append(new_node)
	n2i = {x:i for i, x in enumerate(result)}
	i2n = {i:x for i, x in enumerate(result)}
	# 1st cluster
	# get average neighbor weight
	weight_sum = 0
	neighbor_count = 0
	for node in result:
		assert node in n2i
		for _, weight in node.neighbor:
			weight_sum += weight
			neighbor_count += 1
	avg_weight = weight_sum / neighbor_count
	# cut edges that has little common neighbors
	for node in result:
		# print([y for  _, y in node.neighbor])
		node.neighbor = [(x, y) for x, y in node.neighbor if y > avg_weight and x in n2i]
		print(len(node.neighbor))
	
	wg = []
	for i, node in enumerate(result):
		for n, w in node.neighbor:

			assert n in n2i
			wg.append((i, n2i[n], w))
	graph = nx.MultiGraph()
	graph.add_weighted_edges_from(wg)
	partition = community.best_partition(graph)
	result = []
	print(len(partition))
	for com in set(partition.values()):
		result.append([i2n[n].surface for n in partition.keys() if partition[n] == com])
	return result

class Node():
	def __init__(self, surface, neighbor_entities, source):
		self.surface = surface
		self.source = source
		self.neighbor = set([])
		self.neighbor_entities = set(neighbor_entities)

	def toJSON(self):
		return {
			"surface": self.surface,
			"neighbor": list(self.neighbor_entities)
		}

	def __hash__(self):
		return hash((self.surface, self.source))

if __name__ == '__main__':
	corpus = jsonload("corpus/de_candidates.json")
	cluster = cluster(corpus)
	# jsondump(cluster, "ec_results/louvain_result_only_surface.json")
	jsondump(cluster, "ec_results/louvain_result.json")
	# time_analysis()