from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
import argparse
from peft import PeftModel, PeftConfig

model_path = "./stilts-llm-finetuned/final_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)

print("Loading standard fine-tuned model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
)

# Create streamer
streamer = TextStreamer(
    tokenizer,
)

# Example usage
prompt = "How do I cross-match two astronomical catalogs within 5 arcseconds?"
template = "{% for message in messages %}{{'<|start_header_id|>' + message['role'] + '<|end_header_id|>\n' + message['content'] + '<|eot_id|>' + '\n'}}{% endfor %}"

chat = [
    {
        "role": "system",
        "content": "You are a stilts command generator.",
    },
    {"role": "user", "content": prompt},
]

formatted = tokenizer.apply_chat_template(chat, chat_template=template, tokenize=False)

# add the assistent role header

formatted += "<|start_header_id|>assistant<|end_header_id>\n"
print(formatted)
print("\nGenerating response...")
inputs = tokenizer(
    formatted,
    return_tensors="pt",
).to(model.device)
eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

outputs = model.generate(
    **inputs,
    streamer=streamer,
    max_new_tokens=512,
    eos_token_id=eot_token_id,
    do_sample=True,
    temperature=0.7
)

full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
response = full_output[len(prompt) :]
print("\nResponse:")
print(response)
