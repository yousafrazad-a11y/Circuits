from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
print(tokenizer("Sequence: apple banana mango apple banana")["input_ids"])
print(tokenizer("Sequence: apple banana plum pear kiwi")["input_ids"])
