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

def create_hypergraph_from_dimacs_cnf(num_vars, clauses):
    H = nx.Graph()
    H.add_nodes_from(range(1, num_vars + 1))
    
    for i, clause in enumerate(clauses):
        for var in clause:
            H.add_edge("C_" + str(i), abs(var))
    
    return H

def visualize_hypergraph(H, clauses):
    pos = nx.spring_layout(H)
    # pos = nx.spring_layout(H, k=0.5, iterations=50)   #DO ZMIANY mogą być parametry
    # pos = nx.circular_layout(H) # ciekawy wygląd dla example_1, poza tym nieprzydatny
    # pos = nx.kamada_kawai_layout(H) #ładnie układa dla example_1
    # pos = nx.shell_layout(H) #podobny do circular
    # pos = nx.spiral_layout(H) #spirala

    edge_colors = "black"  # Czarny kolor dla krawędzi
    node_colors = ["lightblue" if "C_" not in str(node) else "yellow" for node in H.nodes]  # Niebieski kolor dla zmiennych
    plt.rc('lines', antialiased=True)
    nx.draw(H, pos, with_labels=True, node_size=300, font_size=10, font_color='black', node_color=node_colors, edge_color=edge_colors)
    plt.title("DIMACS CNF Hypergraph")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a DIMACS CNF file as a hypergraph")
    parser.add_argument("file", help="Path to the DIMACS CNF file")
    args = parser.parse_args()
    
    num_vars, clauses = read_dimacs_cnf(args.file)
    H = create_hypergraph_from_dimacs_cnf(num_vars, clauses)
    visualize_hypergraph(H, clauses)

