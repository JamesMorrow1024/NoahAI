# NoahAI

A machine learning approach for calculating the transaction fees based on the block history. We analyse all transaction history using a network analysis approach to find patterns of relationships between data points and use this relationship together with current operational overhead on the data using read/write/execution counts to train an efficient Neural Network model that can to fast prediction over the price.



This project uses the Feed-Forward Neural Network (FFNN) and Graph Embeddings. The project consists of several Python scripts and modules.
## Basic requirements
1. Python 3.10 or higher
2. Node.js
3. Virtual environment(recommended)
## Basic steps for installation and running
1. Download the training data from the drive and unzip it in the project folder. File path should be looking like project structure.
https://drive.google.com/drive/folders/1rLVkIPmXMxfH5UXkxEusyxKBz6B6tj5_?usp=sharing

3. From the terminal go to the root folder of the project and install requirements then run the model. This code will run the  AI-powered server on port 5000.
```
$pip install -r requirements.txt
$python main.py listen
```
3. Unzip inference.zip
4. Go to the interface folder and run the below code. This will run the interface you can test the model.
```
$npm i
$npm run dev
```
5. When you are at the web interface load the 'meta_data.csv'. This will list all data points and their operations. Then select any one of the data points and hit the calculate button. The calculate button sends a request to AI to calculate the estimated transaction fee.

## Requirements

- setuptools
- Flask
- CORS
- numpy
- gensim
- keras
- matplotlib
- networkx
- node2vec
- pandas
- git+https://github.com/VenkateshwaranB/stellargraph.git
- scikit_learn
- tqdm


## Project Structure
├── main.py  
├── app.py  
├── functions.py  
├── graph_embeddings.py  
├── blocks_parsed  
│   ├── block1  
│   ├── block2  
│   └── ...  
├── graph_embeddings.emb  
├── data.pkl  
├── best_model.keras  
├── meta_data.csv  
├── requirements.txt  
├── README.md  
└── .gitignore  
├── interface  
│   ├── src  
│   └── ...  
## Training and running machine learning

- `main.py`: The main script of the project. It trains the FFNN and Graph Embeddings on a dataset, predicts fees on a test dataset, and saves the results to a CSV file. It also starts a Flask server if the `listen` command is passed.
- `functions.py`: Contains various utility functions used in the project.
- `learning_models.py`: Contains the FFNN and GraphEmbeddings classes.
- `app.py`: Contains the Flask application used in the project.

## Usage

You can run the main script from the command line with the following command:

```
python main.py <command>
```

Replace `<command>` with one of the following commands:

- `train`: Trains the FFNN and Graph Embeddings on the dataset and saves the results to a CSV file.
- `listen`: Starts an inference server on `localhost:5000`.

## Files

## Embedding file

In this project, we use graph embeddings to represent nodes in a graph. Graph embeddings are a way of representing the structure of a graph in a low-dimensional space. They capture the topology of the graph and can be used as input to machine learning algorithms.

The graph embeddings are created using the Node2Vec algorithm, which is an algorithm for learning continuous feature representations for nodes in networks. Node2Vec is a semi-supervised algorithm that generates embeddings for nodes in a graph by optimizing a neighborhood-preserving objective. The objective is to learn a mapping of nodes to a low-dimensional space of features that maximizes the likelihood of preserving network neighbourhoods of nodes.

The Node2Vec model is trained on the graph, and the resulting embeddings are used as input to a Feed-Forward Neural Network (FFNN) for the fee prediction. The Node2Vec model is saved to a file (`graph_embeddings.emb`) and is loaded from this file if it exists.

The dimension of the graph embeddings is set to 64 (`dim = 256`). This means that each node in the graph is represented by a 256-dimensional vector.

The graph embeddings are combined with other features (total cost) to create the final input to the FFNN. The `vectorize_input` function is used to combine the graph embeddings and the total cost into a single vector.

## Data
The `data.pkl` file is a serialized Python object that is used to store the preprocessed data for this project. This file is created using Python's `pickle` module.

The `data.pkl` file contains the following data:

- `node`: A list of nodes in the graph.
- `edge`: A list of edges in the graph.
- `paths`: A list of paths in the graph.

Each node in the `node` list is a dictionary with the following keys:

- `id`: The ID of the node.
- `features`: A list of features for the node.

Each edge in the `edge` list is a dictionary with the following keys:

- `source`: The ID of the source node of the edge.
- `target`: The ID of the target node of the edge.

Each path in the `paths` list is a list of node IDs.

Before running the script, please ensure that the `data.pkl` file is in the correct format and is located in the specified path. Otherwise, the script will generate this file from scratch.

## Training
The training process is initiated by running the `main.py` script with the `train` command. The training process involves the following steps:

1. The script loads the graph embeddings and other features (total cost) from the `data.pkl` file. Each node in the graph is represented by a vector of graph embeddings and the total cost.

2. The data is split into a training set and a test set using a 70-30 split. This means that 70% of the data is used for training the model, and 30% is used for testing the model.

3. The model is trained on the training data using the `train` method of the model. The model is trained for 1000 epochs with a batch size of 4.

4. After training, the model is saved to a file with the name of the model class and the extension `.h5` (for example, `FFNN.h5`).

5. The model is then used to predict the fees on the test data. The results are saved to a CSV file with the name of the model class and the extension `.csv` (for example, `results_FFNN.csv`).

6. Finally, the results are visualized using the `visualize` function.

To start the training process, run the following command:

```bash
python main.py train 
```

## **Inference server**

1. Unzip inference.zip and run the app with npm.
This project includes a server that can be used for inference. The server is implemented in Flask and exposes a `/predict` endpoint that accepts POST requests with the data to predict.

The server is started by running the `main.py` script with the `listen` command. When the server is started, it loads the trained model from the `.h5` file.

The `/predict` endpoint expects a JSON object in the request body with the following format:

```json
{
    "predict": [node_id, s, r, w, x]
}

```
Where:

node_id is the ID of the node for which to predict the fee.
s, r, w, and x are the operational overheads on the data (read/write/execution counts).
The server uses the trained model to predict the fee for the given node and operational overheads. The prediction is returned as a JSON object with the following format:
```
{
    "fee": predicted_fee
}
```
The predicted_fee is the predicted fee for the given node and operational overheads.

To start the server, run the following command:
python main.py listen

## **Future updates**

1. Data clustering for tiered fee detection
2. Training the model using all block information rather than 1 month of history.
3. Crafting more features increase the accuracy of the model.

