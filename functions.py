import matplotlib.pyplot as plt
import networkx as nx
import re
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import pickle
import os
from sklearn.decomposition import PCA

def plot_path(G, start_node):
    # Get all nodes
    nodes = list(G.nodes)

    # Draw the graph
    pos = nx.spring_layout(G)  # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(G, pos, node_color='r')

    # labels
    nx.draw_networkx_labels(G, pos, font_size=10)

    # Iterate over all nodes and draw all paths from start_node to each node
    for node in nodes:
        if node != start_node:
            all_paths = list(nx.all_simple_paths(G, source=start_node, target=node))
            for path in all_paths:
                path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
                nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=2)

    plt.show()
    

def read_block_file(data_path: str, blocks: list[str]) -> tuple[dict, dict, dict]:
    mapping = {"s": 1, "r": 1, "w": 1, "x": 1, "-": 0}
    nodes = {}
    edges = {}
    paths = {}
    df = pd.DataFrame()
    
    meta_dict = {}
    if not os.path.exists("data.pkl"):
        for fil in tqdm(blocks):
            with open(data_path + fil) as file:
                for line in file:
                    pattern_txn = r'\(([^)]+)\)'
                    pattern_fee = r'(\d+\.\d+)(?!.*\d+\.\d+)'

                    matches_txn = re.findall(pattern_txn, line)
                    match_fee = float(re.search(pattern_fee, line).group())
                    # Find matches\
                    path = []
                    for i in range(len(matches_txn)):
                        op, current_node = matches_txn[i].split(", ")
                        ops = [mapping[i] for i in op] 
                        if current_node in meta_dict:
                            meta_dict[current_node]["s"] += ops[0]
                            meta_dict[current_node]["r"] += ops[1]
                            meta_dict[current_node]["w"] += ops[2]
                            meta_dict[current_node]["x"] += ops[3]
                        else:
                            meta_dict[current_node] = {"s": ops[0], "r": ops[1], "w": ops[2], "x": ops[3]}
                            
                        if current_node in nodes:
                            nodes[current_node]["features"][0] += ops[0]
                            nodes[current_node]["features"][1] += ops[1]
                            nodes[current_node]["features"][2] += ops[2]
                            nodes[current_node]["features"][3] += ops[3]
                        else:
                            nodes[current_node] = {"features": ops}
                        path.append(current_node)
                    for p in range(len(path)-1):
                        curr = path[p]
                        next = path[p+1]
                        edges[curr] = next
                    paths[path[0]] = (path[1:], match_fee)
                    
        if not os.path.exists("meta_data.csv"):
            df = df.from_dict(meta_dict, orient="index", columns=["s", "r", "w", "x"])
            df.to_csv("meta_data.csv", index_label="node")
            
        with open('data.pkl', 'wb') as f:
            pickle.dump((nodes, edges, paths), f)
    else:
        with open('data.pkl', 'rb') as f:
            nodes, edges, paths = pickle.load(f)
            
    return nodes, edges, paths    
def populate_graph(nodes: dict, edges: dict):
    G = nx.MultiDiGraph()
    for node, attrs in nodes.items():
        G.add_node(node, **attrs)
    G.add_edges_from(edges.items())
    return G

def visualize(df, model_name):
    # Create a scatter plot
    plt.scatter(df['fee'], df['predicted'])
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted: ' + model_name)

    # Add a line for perfect correlation
    plt.plot([df['fee'].min(), df['fee'].max()], [df['fee'].min(), df['fee'].max()], 'k--')
    # Add a legend
    plt.legend()
    # Show the plot
    plt.show()



def visualize_clusters(data):
    plt.ion()
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(data[['s', 'r', 'w', 'x']])
    principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])

    finalDf = pd.concat([principalDf, data[['cluster']]], axis=1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)

    colors = ['r', 'g', 'b']
    for cluster, color in zip(finalDf['cluster'].unique(), colors):
        indicesToKeep = finalDf['cluster'] == cluster
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'], finalDf.loc[indicesToKeep, 'principal component 2'], c=color, s=50)

    ax.legend(finalDf['cluster'].unique())
    ax.grid()
    plt.draw()
    plt.show(block=True)

        
        
