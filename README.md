
# Semantic Sentence Similarity Model API with Transformers and TensorFlow
This repository implements a semantic sentence similarity model API using Transformers and TensorFlow. The model measures the semantic similarity between two sentences and is based on the STSB-Roberta-Base pre-trained model.

Installation
To get started, you need to install the necessary libraries using pip:
```
!pip install transformers
!pip install sentence-transformers
!pip install torch torchvision -U
```
## Pretrained Model
The pretrained model used in this API is the `stsb-roberta-base` model from the Sentence Transformers library.

Example
Here's a simple example of how to use the model to measure the semantic similarity between two sentences:
```
from sentence_transformers import SentenceTransformer, util

# Load pretrained model stsb-roberta-large
model = SentenceTransformer('stsb-roberta-base')

sentence1 = "I like Python because I can build AI applications"
sentence2 = "I like Python because I can do data analytics"

# Encode sentences to get their embeddings
embedding1 = model.encode(sentence1, convert_to_tensor=True)
embedding2 = model.encode(sentence2, convert_to_tensor=True)

# Compute similarity score of the two embeddings
cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
print("Sentence 1:", sentence1)
print("Sentence 2:", sentence2)
print("Similarity score:", cosine_scores.item())
```

# Data
The model can be trained and tested using a dataset with sentence pairs and their similarity scores. An example dataset is provided in `text_similarity2.csv`

## Training the Semantic Similarity Model
The goal of training is to create a model that can accurately predict the semantic similarity between pairs of sentences. Here's a step-by-step guide on how to achieve this:

1. Prepare the Data
Prepare a dataset with sentence pairs and their corresponding similarity scores. Each pair of sentences should have a similarity score indicating how similar the sentences are (usually a value between 0 and 1, where 1 means highly similar).

2. Data Preprocessing
Preprocess the dataset to prepare it for training. This may involve tokenization, converting sentences to embeddings, and creating input-output pairs for the model.

3. Model Architecture
Choose an appropriate architecture for your model. In this case, you've used a simple feedforward neural network.

4. Compile the Model
Compile the model with an appropriate loss function and optimizer for your task. Since this is a regression problem (predicting similarity scores), mean squared error (MSE) is a common loss function.

```
model.compile(loss='mse', metrics=[tf.keras.metrics.MeanAbsolutePercentageError()], optimizer='adam')
```
5. Train the Model
Split your dataset into training and testing sets. Train the model using the training set and validate it using the testing set.
6. Evaluate the Model
After training, evaluate the model on the test set to measure its performance.
8. Improvements and Fine-Tuning
Based on the evaluation results, make necessary improvements to the model architecture, hyperparameters, or preprocessing techniques to enhance performance.

9. Save the Model
Once you are satisfied with the model's performance, save it for future use.

## API Creation

-Ensure you have the necessary dependencies installed using `pip install tensorflow, fastapi, sentence-transformers, numpy`.
-The provided code loads a pre-trained TensorFlow model `(precily_sns_mod_complete_regression)` and a SentenceTransformer `(stsb-roberta-large)` using caching for efficient performance.
-Run the FastAPI application by executing uvicorn your_script_name:app --reload in the terminal, replacing your_script_name with the name of your Python script (e.g., app.py).
-Access the API by sending a POST request to `http://127.0.0.1:8000/compare` with the sentences to compare.
-Use `curl` or `Python` requests to interact with the API and obtain the similarity score for the given sentences.
-Feel free to tailor this information to better suit your README structure and style.






