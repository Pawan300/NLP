import torch 
from utils import read_data, arrange_data
from constants import model_weight_cross_question, path
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("ktrapeznikov/albert-xlarge-v2-squad-v2")
model = AutoModelForQuestionAnswering.from_pretrained(model_weight_cross_question)

def handle_question(question, passage):
    input_dict = tokenizer.encode_plus(question, passage, return_tensors="pt")
    input_ids = input_dict["input_ids"].tolist()
    start_scores, end_scores = model(**input_dict)

    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    answer = ''.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]).replace('▁', ' ').strip()
    print(answer)

if __name__ == "__main__":
    data = read_data(path)
    data = arrange_data(data)
    passage = "".join(data["passage"])
    handle_question("What did Mike do for a living?", data["passage"].iloc[0])