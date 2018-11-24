# !flask/bin/python
from flask import Flask, jsonify, request, abort, make_response
import json
import pickle as pkl
import tensorflow as tf
from keras.models import model_from_json
from nltk import word_tokenize
from helper import LabelEncoder, Featurizer, build_vocab

app = Flask(__name__)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


def load_model(model_name='models/model_0.json'):
	# load json and create model
	global graph
	graph = tf.get_default_graph()
	model_file = open(model_name)
	model_json = model_file.read()
	model_file.close()
	global model 
	model = model_from_json(model_json)
	# load weights 
	model.load_weights("models/model_0.h5")
	print("Loaded model from disk")
	return model


@app.route('/todo/model/', methods=['POST'])
def run_model():
	if 'headline' not in request.json or 'desc' not in request.json:
		return jsonify({'Error': 'POST Parameter missing headline/desc'})
	desc = request.json['desc']
	desc = word_tokenize(desc)
	headline = request.json['headline']
	headline = word_tokenize(headline)
	vocab = build_vocab()
	n_words = len(vocab)
	with open('data/label_encoder.pkl','rb') as file:
		label_encoder = pkl.load(file)
	featurizer = Featurizer(vocab)
	X = featurizer.featurize([headline+desc])
	with graph.as_default():	
		y_pred = model.predict(X)[0]
	return jsonify({'Category':label_encoder.decode(list(y_pred))})


if __name__ == '__main__':
	load_model()
	app.run(debug=True)