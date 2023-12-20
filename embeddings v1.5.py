import networkx as nx
import matplotlib.pyplot as plt
import argparse
from node2vec import Node2Vec
import itertools
import plotly.graph_objs as go

def read_dimacs_cnf(filename):
    clauses = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith("c"):
                continue
            if line.startswith("p cnf"):
                num_vars, num_clauses = map(int, line.strip().split()[2:])
            else:
                clause = list(map(int, line.strip().split()[:-1]))
                clauses.append(clause)
    return num_vars, clauses

def visualize_graph(clauses):
    G = nx.Graph()
    for clause in clauses:
        for literal in clause:
            G.add_node(abs(literal))  # Dodanie węzłów dla zmiennych
        for i in range(len(clause)):
            for j in range(i+1, len(clause)):
                G.add_edge(abs(clause[i]), abs(clause[j]))  # Dodanie krawędzi między zmiennymi w tej samej klauzuli
    pos = nx.spring_layout(G)  # Ustalenie układu wizualizacji
    nx.draw(G, pos, with_labels=True, node_size=500)
    plt.show()

def generate_node_embeddings(clauses):
    G = nx.Graph()
    for clause in clauses:
        for literal in clause:
            G.add_node(abs(literal))
        for i in range(len(clause)):
            for j in range(i+1, len(clause)):
                G.add_edge(abs(clause[i]), abs(clause[j]))

    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    return model, G

def utworzSkierowanyGraf(G1, model_2):
    G2 = nx.DiGraph()
    for i in G1.nodes:
        node, waga = model_2.wv.most_similar(f'{i}')[0]
        # print(i, int(node), round(waga,2))
        G2.add_edge(i, int(node))  # Dodanie krawędzi z wagą 5
    # G2.add_edge(2, 3, weight=3)  # Dodanie krawędzi z wagą 3
    # G2.add_edge(3, 4, weight=7)  # Dodanie krawędzi z wagą 7

    # Rysowanie grafu
    # pos = nx.shell_layout(G2)
    pos = nx.spring_layout(G2, scale=10)
    # pos = nx.kamada_kawai_layout(G2)
    nx.draw(G2, pos, with_labels=True, node_size=150, arrowsize=20)
    # Dodanie etykiet krawędzi
    labels = nx.get_edge_attributes(G2, 'weight')
    nx.draw_networkx_edge_labels(G2, pos, edge_labels=labels)
    plt.show()

def cala_operacja(nazwa_pliku):
    num_vars, clauses = read_dimacs_cnf(nazwa_pliku)
    model, G = generate_node_embeddings(clauses)
    utworzSkierowanyGraf(G, model)
    return num_vars, clauses

nazwa = "DIMACS_files/turbo_easy/example_2.cnf"
num_vars, clauses = cala_operacja(nazwa)

def interactive_hypergraph(num_vars, clauses, threshold=0):
    G = nx.Graph()

    for var in range(1, num_vars + 1):
        G.add_node(var)

    for i, clause in enumerate(clauses):
        G.add_nodes_from(map(abs, clause))
        for pair in itertools.combinations(map(abs, clause), 2):
            G.add_edge(*pair, clause=i+1)

    nodes_to_remove = [node for node, degree in dict(G.degree()).items() if degree <= threshold]
    G.remove_nodes_from(nodes_to_remove)

    pos = nx.spring_layout(G)

    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)

    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            colorscale='Blues',
            reversescale=False,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Stopień węzła',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)))

    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        node_trace['text'] += ('Var: ' + str(node),)

    for node, adjacencies in enumerate(G.adjacency()):
        node_trace['marker']['color'] += (len(adjacencies[1]),)

    layout = go.Layout(
        titlefont=dict(size=16),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=5, l=5, r=5, t=5),
        annotations=[dict(
            text="",
            showarrow=False,
            xref="paper", yref="paper")],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))

    fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
    fig.show()

interactive_hypergraph(num_vars, clauses, threshold=0)


