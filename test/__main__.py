from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
import argparse
from peft import PeftModel, PeftConfig

# Load the fine-tuned model.
# parser = argparse.ArgumentParser(description="Load and test the fine-tuned model.")

# parser.add_argument(
#     "--model_path",
#     type=str,
#     default="./stilts-llm-finetuned/final_model",
#     help="Path to the fine-tuned model directory.",
# )

# parser.add_argument(
#     "--lora",
#     action="store_true",
#     help="Whether to load the model with LoRA weights.",
# )

# args = parser.parse_args()
# model_path = args.model_path
model_path = "./stilts-llm-finetuned/final_model"
# use_lora = args.lora

# Load tokenizer first to check for special tokens
tokenizer = AutoTokenizer.from_pretrained(model_path)

# if use_lora:
#     print("Loading LoRA adapter...")
#     # Load the exact same base model that was used for fine-tuning
#     base_model = AutoModelForCausalLM.from_pretrained(
#         "meta-llama/Llama-3.2-1B",
#         device_map="auto",
#         torch_dtype=torch.float16,
#     )

#     # Resize token embeddings to match tokenizer
#     base_model.resize_token_embeddings(len(tokenizer))

#     # Then load the LoRA adapter
#     model = PeftModel.from_pretrained(base_model, model_path)
#     model = model.merge_and_unload()
# else:
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
