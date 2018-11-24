# News Category Prediction

## Task : Given the heading and description of a news item, predict the news category

## Example :
	* Heading: What To Watch On Hulu That’s New This Week
	* Description: You're getting a recent Academy Award-winning movie
	* Category : ENTERTAINMENT

## Steps to use API :
1.	In the root folder, run the following commands to setup environment:
```
conda create -n leadiq_ml
conda activate leadiq_ml
pip install -r requirements.txt
```
1.	In the root folder, run the following commands to start the flask server:
```
python app.py
```
1.	With the server running, you can use the following command to predict category:
```
curl -d ‘{“headline”:<headline>,”desc”:<desc>” -H “Content-Type:application/json” -X POST http://127.0.0.1:5000/todo/model/
```

## Format:
*	Input : A json having the following fields:
	*	short_description : string
	*	heading : string
*	Output : A JSON having the following fields:
	*	Category : string
*	Example:
```
curl -d '{"headline":"LOOK: Pope John XXIII Through The Years","desc":""}' -H "Content-Type: application/json"  -X POST http://127.0.0.1:5000/todo/model/
```
Output : { "Category": "RELIGION” }


## Model Info
### Dataset :
* Total of ~125k datapoints
* 80/20 Train-Test split

### Model : 
Deep Learning model using Bag-Of-Words scheme, having the following layers:
*	Input Layer – Count based vector of the input heading + description
*	Dense Layer – Having 100 neurons, Activation Function - ReLU
*	Output Layer – Having 31 neurons, one for each class, using Softmax activation function

### Metrics:
* Number of Epochs in Training : 20
* Train Accuracy : 75.6%
* Test Accuracy : 58.9%
