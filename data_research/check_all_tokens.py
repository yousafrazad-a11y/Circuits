from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
print(tokenizer.encode(" kiwi", add_special_tokens=False))
print(tokenizer.encode(" plum", add_special_tokens=False))
print(tokenizer.encode(" pear", add_special_tokens=False))
