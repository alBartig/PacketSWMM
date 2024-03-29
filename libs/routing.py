import numpy as np
import pandas as pd
import networkx as nx
from swmm_api import read_inp_file, read_out_file
from swmm_api.input_file.macros import inp_to_graph


def table_from_graph(graph, target=None):
    """
    creates a table from the topologically ordered nodes of the input graph
    Args:
        graph (NetworkX.DiGraph): input graph, hydraulic network
        target (str): outlet / target node of the hydraulic network

    Returns:
        pd.DataFrame: table for hydraulic routing of packages
    """
    if target is not None:
        graph = graph.subgraph(list(nx.ancestors(graph, target)))
    columns = list(nx.topological_sort(graph))
    df_routing = pd.DataFrame(columns=columns)
    return df_routing


class Router:
    def __init__(self, path_inp):
        inp = read_inp_file(path_inp)
        self.g_network = inp_to_graph(inp)

    def read_hyraulics(self, path_out):
        out = read_out_file(path_out)
        return flows

    def route_table(self, routing_table):
        df = routing_table
        for node in df.columns:
            succ_node = self.g_network.successors(node)
            succ_edge, succ_length = self.g_network.edges[node, succ_node]["obj"].get("name", "length")
            df.loc[df[node].notnull(), succ_node] = df.loc[df[node].notnull(), node] + timedelta(hours=2)







def main():
    inp_path = r"../sample_data/sample_model.inp"
    router = Router(inp_path)
    pass


if __name__ == "__main__":
    main()
    pass
