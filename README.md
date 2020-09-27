## This repository contains the programs related to NLP.<br>

I have tried some approaches on the simple dataset which trying to classify the types of text into spam or ham.<br>
So I have tried mulitple strategy to come up for the embeddings: <br>

* TF_IDF<br>
* Word2Vec<br>
* Doc2Vec<br>

Then I tried random forest and RNN structure with LSTM.<br>

Scores I get is:<br>
Model          |   Precision  |  Recall   | Accuracy 
-----------------------------------------------------
TF_IDF + RF    |   0.991      |   0.778   |   0.97
Word2Vec + RF  |   0.461      |   0.243   |   0.865
Doc2Vec + RF   |   0.81       |   0.354   |   0.906
