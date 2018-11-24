News Category Prediction

Task : Given the heading and description of a news item, predict the news category

Example :
	Heading: What To Watch On Hulu That’s New This Week
	Description: You're getting a recent Academy Award-winning movie
	Category : ENTERTAINMENT

Steps to use API :
1)	In the root folder, run the following commands to setup environment:
a.	conda create -n leadiq_ml
b.	conda activate leadiq_ml
c.	pip install -r requirements.txt
2)	In the root folder, run the following commands to start the flask server:
a.	python app.py
3)	With the server running, you can use the following command to predict category:
a.	curl -d ‘{“headline”:<headline>,”desc”:<desc>” -H “Content-Type:application/json” -X POST http://127.0.0.1:5000/todo/model/
b.	Input : A json having the following fields:
i.	short_description : string
ii.	heading : string
c.	Output : A JSON having the following fields:
i.	Category : string
d.	Example:
i.	curl -d '{"headline":"LOOK: Pope John XXIII Through The Years","desc":""}' -H "Content-Type: application/json"  -X POST http://127.0.0.1:5000/todo/model/
ii.	Output : { "Category": "RELIGION” }






Model Info
Dataset :
Total of ~125k datapoints
80/20 Train-Test split

Model : 
Deep Learning model using Bag-Of-Words scheme, having the following layers:
1)	Input Layer – Count based vector of the input heading + description
2)	Dense Layer – Having 100 neurons, Activation Function - ReLU
3)	Output Layer – Having 31 neurons, one for each class, using Softmax activation function

Metrics:
	Number of Epochs in Training : 20
	Train Accuracy : 75.6%
	Test Accuracy : 58.9%
