from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
import argparse
from peft import PeftModel, PeftConfig

# Load the fine-tuned model
parser = argparse.ArgumentParser(description="Load and test the fine-tuned model.")
parser.add_argument(
    "--model_path",
    type=str,
    default="./stilts-llm-finetuned/final_model",
    help="Path to the fine-tuned model directory.",
)
parser.add_argument(
    "--lora",
    action="store_true",
    help="Whether to load the model with LoRA weights.",
)

args = parser.parse_args()
model_path = args.model_path
use_lora = args.lora

# Load tokenizer first to check for special tokens
tokenizer = AutoTokenizer.from_pretrained(model_path)

if use_lora:
    print("Loading LoRA adapter...")
    # Load the exact same base model that was used for fine-tuning
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # Resize token embeddings to match tokenizer
    base_model.resize_token_embeddings(len(tokenizer))

    # Then load the LoRA adapter
    model = PeftModel.from_pretrained(base_model, model_path)
    model = model.merge_and_unload()
else:
    print("Loading standard fine-tuned model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )

# Set pad token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Create streamer
streamer = TextStreamer(
    tokenizer, skip_prompt=True, skip_special_tokens=True, handle_stopping_criteria=True
)

# Get model context length
context_length = getattr(model.config, "max_position_embeddings", 2048)
print(f"Model's context window: {context_length} tokens")

max_generation_length = min(1024, context_length - 10)
print(f"Recommended max generation length: {max_generation_length} tokens")


def generate_response(prompt, max_new_tokens=300, temperature=0.7, top_p=0.9):
    """Improved generation function with better stopping criteria"""
    if "### Instruction:" in prompt and not prompt.endswith("\n"):
        prompt += "\n"

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=context_length - 100
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        streamer=streamer,
        pad_token_id=tokenizer.eos_token_id,
    )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_output[len(prompt) :]
    return response.strip()


# Example usage
prompt = "### Instruction: Do sky matching between 'cat1.fits' and 'cat2.fits'.\n### Response:"
print("\nGenerating response...")
response = generate_response(prompt, max_new_tokens=500, temperature=0.8, top_p=0.95)

# print("\nFinal Response:")
# print(response)
