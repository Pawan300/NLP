import os
import re
import json
import string
import numpy as np
import tensorflow as tf
from tensorflow import keras
from constant import max_len
from tensorflow.keras import layers
from tokenizers import BertWordPieceTokenizer

# Load the fast tokenizer from saved file
tokenizer = BertWordPieceTokenizer("bert_base_uncased/vocab.txt", lowercase=True)

class SquadExample:
    def __init__(self, question, context, start_char_idx, answer_text, all_answers):
        self.question = question
        self.context = context
        self.start_char_idx = start_char_idx
        self.answer_text = answer_text
        self.all_answers = all_answers
        self.skip = False

    def preprocess(self):
        context = self.context
        question = self.question
        answer_text = self.answer_text
        start_char_idx = self.start_char_idx

        # Clean context, answer and question
        context = " ".join(str(context).split())
        question = " ".join(str(question).split())
        answer = " ".join(str(answer_text).split())

        # Find end character index of answer in context
        end_char_idx = start_char_idx + len(answer)
        if end_char_idx >= len(context):
            self.skip = True
            return

        # Mark the character indexes in context that are in answer
        is_char_in_ans = [0] * len(context)
        for idx in range(start_char_idx, end_char_idx):
            is_char_in_ans[idx] = 1

        # Tokenize context
        tokenized_context = tokenizer.encode(context)

        # Find tokens that were created from answer characters
        ans_token_idx = []
        for idx, (start, end) in enumerate(tokenized_context.offsets):
            if sum(is_char_in_ans[start:end]) > 0:
                ans_token_idx.append(idx)

        if len(ans_token_idx) == 0:
            self.skip = True
            return

        # Find start and end token index for tokens from answer
        start_token_idx = ans_token_idx[0]
        end_token_idx = ans_token_idx[-1]

        # Tokenize question
        tokenized_question = tokenizer.encode(question)

        # Create inputs
        input_ids = tokenized_context.ids + tokenized_question.ids[1:]
        token_type_ids = [0] * len(tokenized_context.ids) + [1] * len(
            tokenized_question.ids[1:]
        )
        attention_mask = [1] * len(input_ids)

        # Pad and create attention masks.
        # Skip if truncation is needed
        padding_length = max_len - len(input_ids)
        if padding_length > 0:  # pad
            input_ids = input_ids + ([0] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
        elif padding_length < 0:  # skip
            self.skip = True
            return

        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.start_token_idx = start_token_idx
        self.end_token_idx = end_token_idx
        self.context_token_to_char = tokenized_context.offsets


class ExactMatch(keras.callbacks.Callback):
    
    def __init__(self, x_eval, y_eval):
        self.x_eval = x_eval
        self.y_eval = y_eval

    def on_epoch_end(self, epoch, logs=None):
        pred_start, pred_end = self.model.predict(self.x_eval)
        count = 0
        eval_examples_no_skip = [_ for _ in eval_squad_examples if _.skip == False]
        for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
            squad_eg = eval_examples_no_skip[idx]
            offsets = squad_eg.context_token_to_char
            start = np.argmax(start)
            end = np.argmax(end)
            if start >= len(offsets):
                continue
            pred_char_start = offsets[start][0]
            if end < len(offsets):
                pred_char_end = offsets[end][1]
                pred_ans = squad_eg.context[pred_char_start:pred_char_end]
            else:
                pred_ans = squad_eg.context[pred_char_start:]

            normalized_pred_ans = normalize_text(pred_ans)
            normalized_true_ans = [normalize_text(_) for _ in squad_eg.all_answers]
            if normalized_pred_ans in normalized_true_ans:
                count += 1
        acc = count / len(self.y_eval[0])
        print(f"\nepoch={epoch+1}, exact match score={acc:.2f}")

class QnATestData():
    
    def __init__(self):
      self.input_ids = []
      self.token_type_ids = []
      self.attention_masks = []
      self.context_token_to_char = []
        
    def preprocess(self, context, questions):
      input_id = []

      for each_question in questions:

        # Clean context, answer and question
        context = " ".join(str(context).split())
        question = " ".join(str(each_question).split())

        # Tokenize context and question
        tokenized_context = tokenizer.encode(context)
        tokenized_question = tokenizer.encode(each_question)

        # Create inputs
        input_id = tokenized_context.ids + tokenized_question.ids[1:]
        token_type_id = [0] * len(tokenized_context.ids) + [1] * len(tokenized_question.ids[1:])
        attention_mask = [1] * len(input_id)

        # Pad and create attention masks.
        # Skip if truncation is needed
        padding_length = max_len - len(input_id)
        
        if padding_length > 0:  # pad
            input_id = input_id + ([0] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_id = token_type_id + ([0] * padding_length)
        elif padding_length < 0:  # skip
            self.skip = True
            continue
        

        self.input_ids.append(input_id)
        self.token_type_ids.append(token_type_id)
        self.attention_masks.append(attention_mask)
        self.context_token_to_char.append(tokenized_context.offsets)

    def get_test_result(self, context, questions):
      pred_answer_list = []
      self.preprocess(context, questions)
      x = [
        np.array(self.input_ids),
        np.array(self.token_type_ids),
        np.array(self.attention_masks),
      ]

      pred_start, pred_end = model.predict(x)
      for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
        offsets = self.context_token_to_char[idx]
        start = np.argmax(start)
        end = np.argmax(end)
        if start >= len(offsets):
          print("start is greater the offsets")
          continue
        pred_char_start = offsets[start][0]


        if end < len(offsets):
            pred_char_end = offsets[end][1]
            pred_ans = context[pred_char_start:pred_char_end]
        else:
            pred_ans = context[idx][pred_char_start:]
        pred_answer_list.append(pred_ans)
      return pred_answer_list