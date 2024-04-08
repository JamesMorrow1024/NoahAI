# NoahAI

Sure, here's a README that covers the entire project:

```markdown
# StellarGraph Fee Prediction Project

This project uses the StellarGraph library to predict fees using a Feed-Forward Neural Network (FFNN) and Graph Embeddings. The project consists of several Python scripts and modules.

## Requirements

- Python 3
- NumPy
- pandas
- scikit-learn
- Keras
- StellarGraph
- Flask

## Project Structure

- `main.py`: The main script of the project. It trains the FFNN and Graph Embeddings on a dataset, predicts fees on a test dataset, and saves the results to a CSV file. It also starts a Flask server if the `listen` command is passed.
- `functions.py`: Contains various utility functions used in the project.
- `learning_models.py`: Contains the FFNN and GraphEmbeddings classes.
- `app.py`: Contains the Flask application used in the project.

## Usage

You can run the main script from the command line with the following command:

```bash
python main.py <command>
```

Replace `<command>` with one of the following commands:

- `train`: Trains the FFNN and Graph Embeddings on the dataset and saves the results to a CSV file.
- `listen`: Starts a Flask server on `localhost:5000`.

## Data

The script expects the data to be in a specific format. Please ensure your data meets these requirements before running the script.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
```

Please replace the "Data" section with the actual requirements for your data. Also, you might want to add more details about your project, such as how the FFNN and Graph Embeddings are structured, how to use the Flask server, etc.

