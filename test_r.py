# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
# model = AutoModelForCausalLM.from_pretrained(
#     "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# )
# messages = [
#     {"role": "user", "content": "Who are you?"},
# ]


# inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
# outputs = model.generate(
#     inputs, max_new_tokens=200, pad_token_id=tokenizer.eos_token_id
# )

# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# Optional: run on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prompt
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# Use past_key_values for efficiency
output_ids = input_ids
past_key_values = None
max_new_tokens = 500

print(f"\nPrompt: {prompt}")
print("Generated:", end=" ", flush=True)

for _ in range(max_new_tokens):
    with torch.no_grad():
        outputs = model(
            input_ids=output_ids[:, -1:],
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = outputs.logits
        past_key_values = outputs.past_key_values

        next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
        output_ids = torch.cat([output_ids, next_token_id.unsqueeze(-1)], dim=-1)

        next_token = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
        print(next_token, end="", flush=True)

print()
