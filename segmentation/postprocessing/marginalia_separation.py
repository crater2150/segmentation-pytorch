from typing import List, Tuple, Set

from dataclasses import dataclass

import numpy as np

from segmentation.postprocessing.layout_analysis import Baseline
from segmentation.postprocessing.baselines_util import make_baseline_continous
from segmentation.postprocessing.baseline_graph import BaselineGraph, BaselineGraphNode, LabeledLine
from segmentation.postprocessing.layout_line_segment import make_toplines
from segmentation.preprocessing.source_image import SourceImage

import networkx as nx

def bl_graph_to_nx_dag(blg: BaselineGraph):
    topo_digraph = nx.DiGraph()
    for node in blg.nodes:
        topo_digraph.add_node(node.label)
        for child in node.get_below_label_set():
            topo_digraph.add_edge(node.label, child)
    return topo_digraph


@dataclass
class MainTextBodyParameters:
    min_col_width: float = 0.22
    simple_block_detection: bool = True
    simple_block_min_lines: int = 3
    simple_block_max_std_dev = 0.02


@dataclass
class MainTextBody:
    lx1: int
    lx2: int
    rx1: int
    rx2: int

class MainTextBodyDetector:
    def __init__(self, baselines: List[List[Tuple[int,int]]], scaled_image: SourceImage, params: MainTextBodyParameters = MainTextBodyParameters()):
        extruded = make_toplines(baselines, scaled_image)
        self.extruded = extruded

        bl_graph = BaselineGraph.build_graph(extruded.get_baselines(),extruded.get_toplines(),scaled_image.get_width(), scaled_image.get_height())
        bl_graph.visualize(scaled_image.array())

        topo_dg = bl_graph_to_nx_dag(bl_graph)
        trans_red:nx.DiGraph = nx.algorithms.transitive_reduction(topo_dg)

        new_nodes = []
        for node in bl_graph.nodes:
            new_nodes.append(BaselineGraphNode(node.baseline,topline=node.topline,label=node.label,above=[], below=[]))

        for e in trans_red.edges:
            new_nodes[e[0]-1].below.append(new_nodes[e[1]-1])
            new_nodes[e[1] - 1].above.append(new_nodes[e[0] - 1])

        new_bl_graph = BaselineGraph(new_nodes, bl_graph.baseline_acc, bl_graph.topline_acc)
        new_bl_graph.visualize(scaled_image.array())
        self.trans_red = trans_red
        self.trans_red_bl_graph = new_bl_graph
        self.bl_graph = bl_graph
        self.params = params
        self.scaled_image = scaled_image


    def simple_main_text_body(self):
        # detect a main text body using a simple heuristic
        # sort the graph in topological order
        # find lines which have only one outgoing edge

        @dataclass
        class TextBlockChain:
            node_labels: List
            widths: List
            node_labels_set: Set



        topo_sort = list(nx.algorithms.topological_sort(self.trans_red))

        # find the longost block of nodes which only have 1 outgoing line

        longest_chain_widths = []
        text_block_chains: List[TextBlockChain] = []
        advance = 0
        for i, node in enumerate(topo_sort):
            if advance > 0:
                advance -= 1
                continue
            longest_chain = []
            for ix in range(i+1, len(topo_sort)):
                ix_node = topo_sort[ix]
                if len(self.trans_red_bl_graph.nodes[ix_node - 1].below) > 1 and \
                        True: #len(self.trans_red_bl_graph.nodes[ix_node - 1].above) > 1:
                    break
                else:
                    if (ix - i) > len(longest_chain):
                        # check if the chain supports the criteria
                        # calculate length stddev
                        chain = list(topo_sort[i:ix+1])
                        widths = [abs(self.trans_red_bl_graph.nodes[ci-1].baseline.points[0][0] - \
                                  self.trans_red_bl_graph.nodes[ci-1].baseline.points[-1][0]) for ci in chain]
                        if np.std(widths) < self.params.simple_block_max_std_dev * sum(widths) / len(widths):
                            longest_chain = chain
                            longest_chain_widths = widths
                            advance = (ix-i)+1
                            text_block_chains.append(TextBlockChain(node_labels=chain,widths=widths, node_labels_set=set(chain)))

        # plot them

        longest_chain_nodes = set()
        """
        lc_stddev = np.std(longest_chain_widths)
        lc_mean = np.mean(longest_chain_widths)
        for lcn, lcw in zip(longest_chain, longest_chain_widths):
            if abs(lcw - lc_mean) / lc_stddev > 1.0:
                continue
            else:
                longest_chain_nodes.add(lcn)
        """

        # only take nodes which lie in the middle of the longest chain
        for text_block in text_block_chains:
            if len(text_block.node_labels) < self.params.simple_block_min_lines: continue
            lc_stddev = np.std(text_block.widths)
            lc_mean = np.mean(text_block.widths)
            for lcn, lcw in zip(text_block.node_labels, text_block.widths):
                if abs(lcw - lc_mean) / lc_stddev > 1.0:
                    continue
                else:
                    longest_chain_nodes.add(lcn)

        for i in self.trans_red_bl_graph.nodes:
            if i.label not in longest_chain_nodes:
                i.baseline = LabeledLine(points=[],label=i.baseline.label)

        self.trans_red_bl_graph.visualize(self.scaled_image.array(), only_baseline=True)

        # calculate the outer bounds
        return

        # TODO: this doesn't work as expected for column splits, but it's okay

        lp = [self.trans_red_bl_graph.nodes[i-1].baseline[0][0] for i in longest_chain]
        rp = [self.trans_red_bl_graph.nodes[i-1].baseline[-1][0] for i in longest_chain]

        # calculate the median width
        wp = [y-x for x,y in zip(lp, rp)]
        median_width = float(np.median(wp))


