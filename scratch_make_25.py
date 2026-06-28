import re

with open('/home/exouser/pruning/data_research/24_test_dataset3_all_models.py', 'r') as f:
    content = f.read()

new_models = '''MODELS = {
    "qwen_normal_32b": {
        "name": "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8",
        "quant": "gptq"
    },
    "qwen_coder_32b": {
        "name": "Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int8",
        "quant": "gptq"
    },
    "qwen_coder_7b": {
        "name": "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
        "quant": "awq"
    },
    "deepseek_coder_33b": {
        "name": "TheBloke/deepseek-coder-33B-instruct-AWQ",
        "quant": "awq"
    },
    "deepseek_thinking_32b": {
        "name": "RedHatAI/DeepSeek-R1-Distill-Qwen-32B-quantized.w8a8",
        "quant": None
    }
}'''

content = re.sub(r'MODELS = \{[\s\S]*?\n\}', new_models, content)

content = content.replace('dataset3.jsonl', 'dataset4.jsonl')
content = content.replace('dataset3_accuracy_summary.json', 'dataset4_accuracy_summary.json')
content = content.replace('dataset3_failures', 'dataset4_failures')

with open('/home/exouser/pruning/data_research/25_test_dataset4_all_models.py', 'w') as f:
    f.write(content)
