from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
print(tokenizer.encode(" peach", add_special_tokens=False))
print(tokenizer.encode(" peach", add_special_tokens=False)[-1])
