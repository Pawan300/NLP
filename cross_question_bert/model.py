import os
import re
import json
import string
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from Squad_example import SquadExample, ExactMatch
from utils import create_inputs_targets, normalize_text, create_squad_examples, create_model
from transformers import BertTokenizer, TFBertModel, BertConfig
from constant import max_len

configuration = BertConfig()

slow_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
save_path = "bert_base_uncased/"
if not os.path.exists(save_path):
    os.makedirs(save_path)
slow_tokenizer.save_pretrained(save_path)


train_data_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
train_path = keras.utils.get_file("train.json", train_data_url)
eval_data_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
eval_path = keras.utils.get_file("eval.json", eval_data_url)


with open(train_path) as f:
    raw_train_data = json.load(f)

with open(eval_path) as f:
    raw_eval_data = json.load(f)

train_squad_examples = create_squad_examples(raw_train_data)
x_train, y_train = create_inputs_targets(train_squad_examples)
print(f"{len(train_squad_examples)} training points created.")

eval_squad_examples = create_squad_examples(raw_eval_data)
x_eval, y_eval = create_inputs_targets(eval_squad_examples)
print(f"{len(eval_squad_examples)} evaluation points created.")

exact_match_callback = ExactMatch(x_eval, y_eval)

model = create_model()

model.summary()

model.fit(
    x_train,
    y_train,
    epochs=3,
    verbose=2,
    batch_size=64,
    callbacks=[exact_match_callback],
)

context = '''Mike and Morris lived in the same village. While Morris owned the largest jewelry shop in the village, Mike was a poor farmer. Both had large families with many sons, daughters-in-law and grandchildren. One fine day, Mike, tired of not being able to feed his family, decided to leave the village and move to the city where he was certain to earn enough to feed everyone. Along with his family, he left the village for the city. At night, they stopped under a large tree. There was a stream running nearby where they could freshen up themselves. He told his sons to clear the area below the tree, he told his wife to fetch water and he instructed his daughters-in-law to make up the fire and started cutting wood from the tree himself. They didn’t know that in the branches of the tree, there was a thief hiding. He watched as Mike’s family worked together and also noticed that they had nothing to cook. Mike’s wife also thought the same and asked her husband ” Everything is ready but what shall we eat?”. Mike raised his h '''
questions = ["What did Morris owned?", "What did Mike do for a living?", "what the instruction he gave to daughters-in-law? "]
qna_test_obj = QnATestData()
qna_test_obj.get_test_result(context, questions)

