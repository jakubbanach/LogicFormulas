import networkx as nx
import matplotlib.pyplot as plt
import argparse

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

def create_graph_from_dimacs_cnf(num_vars, clauses):
    G = nx.Graph()
    G.add_nodes_from(range(1, num_vars + 1))
    for clause in clauses:
        for i in range(len(clause)):
            for j in range(i + 1, len(clause)):
                G.add_edge(abs(clause[i]), abs(clause[j]))
    return G

def visualize_graph(G):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=300, node_color='skyblue', font_size=10, font_color='black')
    plt.title("DIMACS CNF Graph")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a DIMACS CNF file as a graph")
    parser.add_argument("file", help="Path to the DIMACS CNF file")
    args = parser.parse_args()
    
    num_vars, clauses = read_dimacs_cnf(args.file)
    G = create_graph_from_dimacs_cnf(num_vars, clauses)
    visualize_graph(G)
