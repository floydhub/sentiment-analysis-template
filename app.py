import os
import numpy as np
import pickle

import flask
from flask import Flask
from flask import Flask, request

import tensorflow as tf

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from serving import load_pretrained_model
"""
Import all the dependencies you need to load the model,
preprocess your request and postprocess your result
"""
LABELS = ['negative', 'positive']

# Hyperparams if GPU is available
if tf.test.is_gpu_available():
    # GPU
    MAX_LEN = 500 # Max length of review (in words)
# Hyperparams for CPU training
else:
    # CPU
    MAX_LEN = 90

# Model Parameters
EMBEDDING_DIM = 40
NUM_FILTERS = 250
KERNEL_SIZE = 3
HIDDEN_DIMS = 250

app = Flask(__name__)
app.config['DEBUG'] = False

# MODELS
model = None

# Tokenizer
tokenizer = None

# PATHS
TOKENIZER_PATH = '/models/tokenizer.pickle'
MODEL_PATH = '/models/cnn_sentiment_weights.h5'

def load_model():
	"""Load the model"""
	global model, tokenizer
	model = load_pretrained_model(MODEL_PATH,
								tokenizer.num_words,
								MAX_LEN,
								EMBEDDING_DIM,
								NUM_FILTERS,
								KERNEL_SIZE,
								HIDDEN_DIMS)

def load_tokenizer():
	"""Load the tokenizer"""
	global tokenizer
	# loading
	with open(TOKENIZER_PATH, 'rb') as handle:
		tokenizer = pickle.load(handle)

def data_preprocessing(review):
	"""From text to tokens"""
	global tokenizer

	# Tokenization and Padding
	review_np_array = tokenizer.texts_to_sequences([review])
	review_np_array = sequence.pad_sequences(review_np_array, maxlen=MAX_LEN, padding="post", value=0)
	return review_np_array

# Every incoming POST request will run the `evaluate` method
# The request method is POST (this method enables your to send
# arbitrary data to the endpoint in the request body,
# including images, JSON, encoded-data, etc.)
@app.route('/<path:path>', methods=["POST"])
def evaluate(path):
	""""Preprocessing the data and evaluate the model"""
	if flask.request.method == "POST":

		review = request.form.get("text")
		review_np_array = data_preprocessing(review)

		global model

		# Test model
		score = model.predict(review_np_array)[0][0]
		prediction = LABELS[model.predict_classes(review_np_array)[0][0]]
		output = 'REVIEW: {}\nPREDICTION: {}\nSCORE: {}\n'.format(review, prediction, score)
		return output

# Load the model and run the server
if __name__ == "__main__":
	print(("* Loading model and starting Flask server..."
		"please wait until server has fully started"))
	load_tokenizer()
	load_model()
	app.run(host='0.0.0.0', threaded=False)