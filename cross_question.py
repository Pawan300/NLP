import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("ktrapeznikov/albert-xlarge-v2-squad-v2")
model = AutoModelForQuestionAnswering.from_pretrained(
    "ktrapeznikov/albert-xlarge-v2-squad-v2"
)


def handle_question(question, passage):
    input_dict = tokenizer.encode_plus(question, passage, return_tensors="pt")
    input_ids = input_dict["input_ids"].tolist()
    result = model(**input_dict)

    start_scores = result["start_logits"]
    end_scores = result["end_logits"]

    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    answer = (
        "".join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores) + 1])
        .replace("▁", " ")
        .strip()
    )
    print(answer)


if __name__ == "__main__":
    passage = """
    Mike and Morris lived in the same village. While Morris owned the largest jewelry shop in the village, Mike was a poor farmer. Both had large families with many sons, daughters-in-law and grandchildren. One fine day, Mike, tired of not being able to feed his family, decided to leave the village and move to the city where he was certain to earn enough to feed everyone. Along with his family, he left the village for the city. At night, they stopped under a large tree. There was a stream running nearby where they could freshen up themselves. He told his sons to clear the area below the tree, he told his wife to fetch water and he instructed his daughters-in-law to make up the fire and started cutting wood from the tree himself. They didn’t know that in the branches of the tree, there was a thief hiding. He watched as Mike’s family worked together and also noticed that they had nothing to cook. Mike’s wife also thought the same and asked her husband ” Everything is ready but what shall we eat?”. Mike raised his hands to heaven and said ” Don’t worry. He is watching all of this from above. He will help us.”
The thief got worried as he had seen that the family was large and worked well together. Taking advantage of the fact that they did not know he was hiding in the branches, he decided to make a quick escape. He climbed down safely when they were not looking and ran for his life. But, he left behind the bundle of stolen jewels and money which dropped into Mike’s lap. Mike opened it and jumped with joy when he saw the contents. The family gathered all their belongings and returned to the village. There was great excitement when they told everyone how they got rich.
    """
    handle_question("What did Mike do for a living?", passage)
