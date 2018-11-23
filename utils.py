import numpy as np

def build_vocab(filename='data/vocab.txt'):
	""" Construct Vocab Dictionary from input file """
	vocab = {}
	count = 0
	for line in open(filename):
		vocab[line.strip()] = count
		count += 1
	return vocab

class Featurizer:
	""" Converts the given sentence to respective Bag-Of-Words Vector """
	def __init__(self, vocab):
		self.vocab = vocab

	def featurize(self, _X):
		X = np.zeros((len(_X),len(self.vocab)))
		for xidx, _x in enumerate(_X):	
			for token in _x:
				token = token.lower()
				if token in self.vocab:
					idx = self.vocab[token]
					X[xidx][idx] += 1
		return X



class LabelEncoder:
	""" Converts the given input to encoded labels """
	def __init__(self, y):
		self.encoder = {k:idx for idx,k in enumerate(set(y))}
		self.decoder = {idx:k for k,idx in self.encoder.iteritems()}

	def encode(self, _y):
		y = np.zeros((len(_y),self.num_classes()))
		for item_idx, item in enumerate(_y):
			idx = self.encoder[item]
			y[item_idx][idx] = 1
		return y

	def decode(self, y_pred):
		y_pred_idx = y_pred.index(max(y_pred))
		return self.decoder[y_pred_idx]
	
	def num_classes(self):
		return len(self.encoder)