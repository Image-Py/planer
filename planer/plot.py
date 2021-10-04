import networkx as nx
from matplotlib import pyplot as plt

def plot_net(inputs, inits, body, flow):
    oplst, inputlst, rstlst, others = [], [], [], []
    inout = inputs + ['plrst']
    G = nx.DiGraph()
    for ins, ops, ots in flow:
        if not isinstance(ins, list): ins = [ins]
        if not isinstance(ots, list): ots = [ots]
        for i in ins:
            G.add_node(i); others.append(i)
        G.add_node(ops[0]); oplst.append(ops[0])
        for i in ots: G.add_node(i)
        for i in ins:
            G.add_edge(i, ops[0]); others.append(i)
        for i in ots: G.add_edge(ops[0], i)

    G.remove_nodes_from(inits)
    
    pos = nx.kamada_kawai_layout(G)

    opnode = set(oplst)-set(inout)-set(inits)
    subnode = set(others)-set(inout)-set(inits)
    
    nx.draw_networkx_nodes(G, pos, nodelist=opnode, node_shape='p', node_color='green')
    nx.draw_networkx_nodes(G, pos, nodelist=subnode, node_shape='o', node_color='orange')
    nx.draw_networkx_nodes(G, pos, nodelist=set(inout), node_shape='o', node_color='red')
    
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=10)
    nx.draw_networkx_labels(G, pos, font_size=8)

    import matplotlib.pyplot as plt
    plt.axis('off')
    plt.show()
