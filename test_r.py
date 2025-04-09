from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
)
messages = [
    {"role": "user", "content": "Who are you?"},
]


inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
outputs = model.generate(
    inputs, max_new_tokens=200, pad_token_id=tokenizer.eos_token_id
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
