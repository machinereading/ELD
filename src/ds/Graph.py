from ..utils import readfile
from typing import Dict, Set
class Graph:
	def __init__(self):
		self.nodes: Dict[str, Node] = {}
		self.edges: Set[Edge] = set([])

	def add_node(self, name):
		if name in self.nodes:
			return
		self.nodes[name] = Node(name)

	def add_edge(self, from_name, relation, to_name):
		edge = Edge(from_name, relation, to_name)
		if edge in self.edges: return
		self.nodes[from_name].add_edge(edge)
		self.nodes[to_name].add_edge(edge)
		self.edges.add(edge)

	def __getitem__(self, item):
		if type(item) is str:
			return self.nodes[item]
		e1, e2 = item
		return self.nodes[e1][e2]

	def add_kb_file(self, graph, add_new_node=True):
		object_prefix = "http://ko.dbpedia.org/resource/"
		property_prefix = "http://ko.dbpedia.org/property/"
		if type(graph) is str:
			graph = readfile(graph)
		for triple in graph:
			s, p, o, _ = triple.split("\t")
			s = s.strip("<>").replace(object_prefix, "")
			p = p.strip("<>").replace(property_prefix, "")
			o = o.strip("<>").replace(object_prefix, "")
			if not add_new_node and (s not in self.nodes and o not in self.nodes): continue
			self.add_node(s)
			self.add_node(o)
			self.add_edge(s, p, o)



class Node:
	def __init__(self, name):
		self.name = name
		self.incoming_edges: Dict[str, Set[Edge]] = {}
		self.outgoing_edges: Dict[str, Set[Edge]] = {}

	def __getitem__(self, name):
		incoming = self.incoming_edges[name] if name in self.incoming_edges else []
		outgoing = self.outgoing_edges[name] if name in self.outgoing_edges else []
		return incoming, outgoing

	def add_edge(self, edge):
		f = edge.e1
		t = edge.e2
		
		if f == self.name:
			if f not in self.outgoing_edges:
				self.outgoing_edges[f] = set([])
			self.outgoing_edges[f].add(edge)
		elif t == self.name:
			if t not in self.incoming_edges:
				self.incoming_edges[t] = set([])
			self.incoming_edges[t].add(edge)

	def __str__(self):
		result = [self.name]
		result.append("Incoming:")
		for x in self.incoming_edges.values():
			for item in x:
				result.append(str(item))
		result.append("Outgoing:")
		for x in self.outgoing_edges.values():
			for item in x:
				result.append(str(item))
		return "\n".join(result)

class Edge:
	def __init__(self, e1, r, e2):
		self.e1 = e1
		self.r = r
		self.e2 = e2

	def __eq__(self, obj):
		if type(obj) is not Edge:
			return False
		return obj.e1 == self.e1 and obj.r == self.r and obj.e2 == self.e2

	def __ne__(self, obj):
		return not self.__eq__(obj)

	def __hash__(self):
		return hash((self.e1, self.e2, self.r))

	def __str__(self):
		return "\t".join([self.e1, self.r, self.e2])
