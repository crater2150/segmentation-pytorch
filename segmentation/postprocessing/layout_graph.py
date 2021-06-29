from dataclasses import dataclass, field
from typing import List, Set, Tuple, Dict,
from collections import defaultdict
from functools import partial



@dataclass
class LayoutGraphNode:
    id: int
    data: object

    def __hash__(self):
        return self.id

@dataclass
class LayoutGraphEdge:
    pass


class LayoutGraph:
    #nodes: List[LayoutGraphNode]
    #edges: Dict[LayoutGraphNode, Set[LayoutGraphNode]] = field(default_factory=partial(defaultdict,set))
    __slots__ = ["nodes", "edges"]

    def __init__(self):
        self.nodes: List[LayoutGraphNode] = []
        self.edges: Dict[LayoutGraphNode, Set[LayoutGraphNode]] = defaultdict(set)

    def add_edge(self, n1: LayoutGraphNode, n2: LayoutGraphNode):
        self.edges[n1].add(n2)
        self.edges[n2].add(n1)

    def remove_edge(self, n1: LayoutGraphNode, n2:LayoutGraphNode):
        try:
            self.edges[n1].remove(n2)
            self.edges[n2].remove(n1)
        except KeyError:
            pass

    def remove_node(self, node: LayoutGraphNode):
        self.nodes[node.id] = None
        for tgt in self.edges[node]:
            self.remove_edge(node,tgt)
        self.edges.pop(node)

    def __iter__(self):
        return filter(lambda x: x is not None, self.nodes)

    def iter_edges(self):
        for n in self:
            yield from self.edges[n]

    def add_node(self):
        node = LayoutGraphNode(id=len(self.nodes), data=None)
        self.nodes.append(node)
        return node

    def split_node(self, node: LayoutGraphNode):
        new_node = self.add_node()
        for e in self.edges[node]:
            self.add_edge(new_node,e)








