from ...utils import jsonload, jsondump
from ...utils.TimeUtil import TimeChecker, time_analysis
class StatisticClustering():
	def __init__(self):
		pass

	def cluster(self, corpus):
		with TimeChecker("Graph Generation"):
			graph = self.generate_graph(corpus)

		with TimeChecker("1st clustering"):
			# 1st cluster
			# get average neighbor weight
			weight_sum = 0
			neighbor_count = 0
			for node in graph:
				for _, weight in node.neighbor:
					weight_sum += weight
					neighbor_count += 1
			avg_weight = weight_sum / neighbor_count
			# cut edges that has little common neighbors
			for node in graph:
				# print([y for  _, y in node.neighbor])
				node.neighbor = [(x, y) for x, y in node.neighbor if y > avg_weight]

			# cluster
			clusters = []
			while len(graph) > 0:
				next_node = graph.pop()
				queue = [next_node]
				trash = []
				cluster = set([])
				while len(queue) > 0:
					print(len(queue), len(graph))
					n = queue.pop(0)
					if n in trash: continue
					cluster.add(n)
					new_elems = [x for x, _ in n.neighbor if x not in cluster]
					queue += new_elems

					trash.append(n)
				graph = [x for x in graph if x not in cluster]
				# print(len(node.neighbor), len(cluster), len(graph))
				clusters.append(cluster)

		with TimeChecker("postprocessing"):
			result = []
			for cluster in clusters:
				c = []
				for node in cluster:
					c.append(node.surface)
				result.append(c)
		return result

	def generate_graph(self, corpus):
		result = []

		for item in corpus:
			add_flag = True
			new_node = Node(item["surface"], item["neighbors"], item["fileName"])
			for node in result:
				if new_node.source == node.source:
					if new_node.surface == node.surface:
						node.neighbor = node.neighbor.union(new_node.neighbor)
						add_flag = False
				dup = len(node.neighbor_entities.intersection(new_node.neighbor_entities))
				if dup > 0:
					# print(dup)
					node.neighbor.add((new_node, dup))
					new_node.neighbor.add((node, dup))
			if add_flag:
				result.append(new_node)
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


if __name__ == '__main__':
	sc = StatisticClustering()
	corpus = jsonload("corpus/de_candidates.json")
	cluster = sc.cluster(corpus)
	jsondump(cluster, "ec_results/StatisticClusteringResult2.json")
	time_analysis()