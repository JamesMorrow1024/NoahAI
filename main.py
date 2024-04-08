
import argparse
import os

from sklearn.model_selection import train_test_split
from functions import *
from stellargraph import StellarGraph as sg
import numpy as np
import pandas as pd
from learning_models import FFNN, GraphEmbeddings, Clustering
from app import app


PATH  = "./block_parsed/"
embedding_path = "./graph_embeddings.emb"

def main():
    blocks = os.listdir(PATH)
    node, edge, paths = read_block_file(PATH, blocks)
    print(f"Length of node: {len(node)}")
    print(f"Length of edge: {len(edge)}")
    print(f"Length of paths: {len(paths)}")
    G = populate_graph(node, edge)
    g_feature_attr = G.copy()
    dim = 64
    model = FFNN(input_dim=dim+4)
    embedding_model = GraphEmbeddings()
    if not os.path.exists(embedding_path):
        embedding_model.train_node2vec(graph=g_feature_attr, dimensions=dim)
        embedding_model.node2vecmodel.save(embedding_path)
    else:
        embedding_model.load_model(embedding_path)
    model.embedding_layer = embedding_model.node2vecmodel 
    print(f"Graph embedding dimension: {len(embedding_model.node2vecmodel)}")
    data_x = []
    data_y = []
    for i in paths.items():
        fee_payer = i[0]
        path = i[1][0]
        embedding_model.node2vecmodel[fee_payer]
        total_cost = np.array([0,0,0,0])
        for y in path:
            node_cost = node[y]["features"]
            total_cost = np.sum((node_cost, total_cost), axis=0)
        embeddings = model.vectorize_input(fee_payer, total_cost)
        fee = i[1][1]
        data_x.append(embeddings)
        data_y.append(fee)
        
    X_train, X_test, y_train, y_test = train_test_split(np.array(data_x), np.array(data_y), train_size=0.7)
    print(
    "Array shapes:\n X_train = {}\n y_train = {}\n X_test = {}\n y_test = {}".format(
        X_train.shape, y_train.shape, X_test.shape, y_test.shape
        )
    )

    
    model.train(X_train, y_train, X_test, y_test, epochs=100, batch_size=8)
    results = model.predict(X_test)
    df = pd.DataFrame(list(zip(y_test, results)), columns=["fee", "predicted"])
    df.to_csv(f'results_{model.__class__.__name__}.csv')

    
def analysis():
    df = pd.read_csv("clusters.csv")
    visualize_clusters(df)
    df = pd.read_csv("results_ffnn.csv")
    visualize(df, "model")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="train or listen")
    args = parser.parse_args()
    if args.command == "train":
        main()
    elif args.command == "listen":
        app.run(host='localhost', port=5000, debug=True)
    elif args.command == "analysis":
        analysis()