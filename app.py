from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
from flask_cors import CORS
import os
from learning_models import FFNN, GraphEmbeddings
app = Flask(__name__)
CORS(app)
embedding_model = GraphEmbeddings()
embedding_path = "./graph_embeddings.emb"
# model_path = "FFNN.h5" 
model_path = "best_model.keras" 
model = FFNN(input_dim=68)

try:
    model.load_model(model_path)
except OSError:
    print("Model file does not exist. Please train the model.")
   
try:
    embedding_model.load_model(embedding_path)
except OSError:
    print("Embedding file does not exist. Please train the embeddings.")
    
    
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from POST request
    data = request.get_json(force=True)
    datum = data['predict']
    
    fee_payer = np.array(embedding_model.node2vecmodel[datum[0]])
    
    operations = np.array([int(i) for i in datum[1:]])
    print(fee_payer.shape)
    if model is None:
        return jsonify({"error": "Model is not trained. Please train the model."}), 500
    model_input = np.concatenate((fee_payer, operations))
    # Make prediction using model loaded from disk as per the data
    prediction = model.predict(model_input.reshape(1, -1) )

    # Take the first value of prediction
    output = prediction[0]

    return jsonify(abs(output.item()))


if __name__ == "__main__":
    app.run(port=5000, debug=True)