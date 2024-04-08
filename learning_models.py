from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.linear_model import LinearRegression
from node2vec import Node2Vec
from gensim.models import KeyedVectors
import numpy as np
from keras.callbacks import ModelCheckpoint
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class GraphEmbeddings():
    def __init__(self):
        self.graph_stellar = None
        self.node2vecmodel = None

    def train_node2vec(self, graph, dimensions=256, walk_length=16, num_walks=2, window=8):
        node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks)
        model = node2vec.fit(window=window, min_count=1)
        self.node2vecmodel = model.wv  # Save the KeyedVectors object for later use

    def save_model(self, file_name):
        self.node2vecmodel.save(file_name)

    def load_model(self, file_name):
        self.node2vecmodel = KeyedVectors.load(file_name)

class FFNN:
    def __init__(self, input_dim=132, acttivation='tanh') -> None:
        self.embedding_layer = None
        self.model = Sequential()
        self.model.add(Dense(260, input_dim=input_dim, activation=acttivation))
        self.model.add(Dense(1, activation='tanh'))
        self.model.compile(loss='mean_squared_error', optimizer=Adam())

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[checkpoint])
        
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def vectorize_input(self, fee_payer, total_cost):
        graph_embeddings = self.embedding_layer[fee_payer]
        return np.concatenate((graph_embeddings, total_cost)) 
    
    def save_model(self, file_name):
        self.model.save(file_name)

    def load_model(self, file_name):
        self.model = load_model(file_name)
    
class Regression:
    def __init__(self):
        self.model = LinearRegression()
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    def predict(self, X_test):
        results = self.model.predict(X_test)
        
class Clustering:
    def __init__(self, n_clusters=4):
        self.model = KMeans(n_clusters=n_clusters)

    def train(self, X_train):
        self.model.fit(X_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def find_optimal_clusters(self, data, max_clusters):
        inertias = []
        for i in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, random_state=0).fit(data)
            inertias.append(kmeans.inertia_)
        plt.plot(range(1, max_clusters + 1), inertias, 'bx-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.title('The Elbow Method showing the optimal number of clusters')
        plt.show()
        
    def find_and_assign(self, data):
        self.train(data[['s', 'r', 'w', 'x']])
        data['cluster'] = self.predict(data[['s', 'r', 'w', 'x']])
        data.to_csv("clusters.csv", orient='index')
        
