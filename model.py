from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import pickle as pkl
import random
import numpy as np
from keras.models import model_from_json

from helper import LabelEncoder, Featurizer, build_vocab


def train_test_split(X, y, fraction):
	""" Split the dataset into train/test splits """
	data = zip(X,y)
	n_train = int(fraction * len(data))
	# Shuffle the data to avoid biased splits 
	random.shuffle(data)
	X = map(lambda x: x[0], data)
	y = map(lambda x: x[1], data)
	X_train = np.array(X[:n_train])
	X_test = np.array(X[n_train:])
	y_train = np.array(y[:n_train])
	y_test = np.array(y[n_train:])
	return X_train, y_train, X_test, y_test


vocab = build_vocab()
n_words = len(vocab)
_X, _y = pkl.load(open('data/dataset.pkl'))

print "Dataset loaded"

label_encoder = LabelEncoder(_y)
pkl.dump(label_encoder, open('data/label_encoder.pkl','wb'))
y = label_encoder.encode(_y)

featurizer = Featurizer(vocab)
X = featurizer.featurize(_X)

X_train, y_train, X_test, y_test = train_test_split(X, y, fraction=0.8)
print "Train / Test split constructed"

print X_train.shape, y_train.shape, X_test.shape, y_test.shape

# Model Definition
model = Sequential()
model.add(Dense(100, input_dim=n_words, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(label_encoder.num_classes(), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print "Model Compiled"

# Model Training
model.fit(X_train, y_train, epochs=20, verbose=1)
print "Model trained"

# SAVE MODEL
model.save_weights('models/model_0.h5')
json_string = model.to_json()
f = open('models/model_0.json','w')
f.write(json_string)
f.close()

# Model Test Metrics
loss, accuracy = model.evaluate(X_test, y_test)

print "Model Accuracy : %.2f" % (accuracy*100)





