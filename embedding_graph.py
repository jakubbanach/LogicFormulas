import networkx as nx
import matplotlib.pyplot as plt
import argparse
from node2vec import Node2Vec

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
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a DIMACS CNF file as an embedding")
    parser.add_argument("file", help="Path to the DIMACS CNF file")
    args = parser.parse_args()
    
    num_vars, clauses = read_dimacs_cnf(args.file)
    print(num_vars)
    print(clauses.shape)
    # visualize_graph(clauses)
    # node_embeddings_model = generate_node_embeddings(clauses)
    
    # # Get the embeddings for a specific node (e.g., node 1)
    # node_1_embedding = node_embeddings_model.wv.get_vector(str(1))
    # print("Embedding for node 1:", node_1_embedding)
