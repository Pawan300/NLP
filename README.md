## This repository contains the programs related to NLP.<br>

* This contain some research paper implementation or some transformers extension of hugging face in Text similarity.

* I have tried some approaches on the simple dataset which trying to classify the types of text into spam or ham.<br>
  So I have tried mulitple strategy to come up for the embeddings: <br>

   * TF_IDF<br>
   * Word2Vec<br>
   * Doc2Vec<br>

   Then I tried random forest and RNN structure with LSTM.<br>

   Scores I get is:<br>
   Model                 |   Precision  |  Recall   | Accuracy 
   ----------------------|--------------|-----------|----------
   TF_IDF + RF           |   0.99       |   0.78    |   0.97
   Word2Vec + RF         |   0.46       |   0.24    |   0.87
   Doc2Vec + RF          |   0.81       |   0.35    |   0.91
   RNN + text_to_sequence|   0.99       |   0.96    |   0.99

   I also tried to catch some hyperparameter using different methods and libraries :<br>


   Model                 |  Time (in min)  |  Accuracy 
   ----------------------|-----------------|-----------
   Random forest (RF)    |     2.4         |   0.97
   Grid Search CV        |     25.6        |   0.97    
   Pipeline              |     10.9        |   0.95    
   Skopt                 |     19.3        |   0.97    
   Hyperopt              |     28:12       |   0.95
   Optuna                |     40          |   0.97
