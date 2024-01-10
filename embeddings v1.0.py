import networkx as nx
import matplotlib.pyplot as plt
import argparse
from node2vec import Node2Vec
from sklearn.decomposition import PCA

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

def adding_to_graph(clauses):
    G = nx.Graph()
    for clause in clauses:
        for i in range(len(clause)):
            for j in range(i+1, len(clause)):
                G.add_edge(abs(clause[i]), abs(clause[j]))  # Dodanie krawędzi między zmiennymi w tej samej klauzuli
    return G

def visualize_graph(G):
    pos = nx.spring_layout(G)  # Ustalenie układu wizualizacji
    nx.draw(G, pos, with_labels=True, node_size=500)
    plt.show()

def generate_node_embeddings(G):
    node2vec = Node2Vec(G, dimensions=64, walk_length=40, num_walks=40, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    # node2vec.graph.save("graph.kg")
    return model

def visualize_node_embeddings(model):
    node_ids = model.wv.index_to_key
    print(node_ids)
    node_embeddings = [model.wv[str(node_id)] for node_id in node_ids]

    pca = PCA(n_components=3)
    embeddings_2d = pca.fit_transform(node_embeddings)

    plt.figure(figsize=(8, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
    for i, txt in enumerate(node_ids):
        plt.annotate(txt, (embeddings_2d[i, 0], embeddings_2d[i, 1]), xytext=(5, 2), textcoords='offset points')
    plt.title('Node Embeddings Visualization (2D)')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

# def visualize_node_embeddings(model, graph):
#     node_ids = list(graph.nodes)
#     node_embeddings = [model.get_vector(str(node_id)) for node_id in node_ids]

#     pca = PCA(n_components=2)
#     embeddings_2d = pca.fit_transform(node_embeddings)

#     plt.figure(figsize=(8, 8))
#     plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
#     for i, txt in enumerate(node_ids):
#         plt.annotate(txt, (embeddings_2d[i, 0], embeddings_2d[i, 1]), xytext=(5, 2), textcoords='offset points')
#     plt.title('Node Embeddings Visualization (2D)')
#     plt.xlabel('Component 1')
#     plt.ylabel('Component 2')
#     plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a DIMACS CNF file as an embedding")
    parser.add_argument("file", help="Path to the DIMACS CNF file")
    args = parser.parse_args()
    # G = nx.Graph()
    
    num_vars, clauses = read_dimacs_cnf(args.file)
    G = adding_to_graph(clauses)
    print(num_vars)
    # print(clauses.shape)
    # visualize_graph(G)
    node_embeddings_model = generate_node_embeddings(G)
    # print(node_embeddings_model.wv.most_similar('1'))
    visualize_node_embeddings(node_embeddings_model)
    
    
    # # Get the embeddings for a specific node (e.g., node 1)
    # node_1_embedding = node_embeddings_model.wv.get_vector(str(1))
    # print("Embedding for node 1:", node_1_embedding)
