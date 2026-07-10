from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
print(len(tokenizer.encode("Sequence: apple banana mango apple banana")))
print(len(tokenizer.encode("Sequence: car truck plane car truck")))
print(len(tokenizer.encode("Sequence: orange purple yellow orange purple")))
