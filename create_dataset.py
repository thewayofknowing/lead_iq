import json
from nltk import word_tokenize
from collections import defaultdict
import pickle as pkl
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

# Input : list of news item - short description and heading 
# Output : Dictionary containing word -> count mapping, taking the words with occurrence > 10 
def build_vocab(arg, max_words=5000):
	vocab = defaultdict(int)
	stopwords = open('data/stopwords.txt').read().split('\n')
	for sentence in arg:
		for token in sentence:
			if token.lower() not in stopwords:
				vocab[token.lower()] += 1
	vocab = sorted(vocab.iteritems(), key=lambda x: -x[1])[:max_words]
	vocab = map(lambda x: x[0], vocab)
	return vocab


dataset_filename = "News_Category_Dataset.json"

X, y = [], []

# Read a line of json datapoint at a time and parse it 
for line in open(dataset_filename):
	obj = json.loads(line)
	desc = obj['short_description']
	desc = word_tokenize(desc)
	headline = obj['headline']
	headline = word_tokenize(headline)
	X.append((desc + headline))
	category = obj['category']
	y.append(category)

vocab = build_vocab(X)

# Write to Vocab File
f = open('data/vocab.txt','w')
f.write('\n'.join(vocab))
f.close()

print "Vocab Construction Complete"

print len(X), len(y), len(vocab)

pkl.dump((X,y), open('data/dataset.pkl','wb'))
print "Dataset Dumped"

