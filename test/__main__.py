from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch

# Load the fine-tuned model
model_path = "./stilts-llm-finetuned/final_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

# Set pad token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Create streamer
streamer = TextStreamer(
    tokenizer, skip_prompt=True, skip_special_tokens=True, handle_stopping_criteria=True
)

# Get model context length (not necessarily max generation length)
context_length = getattr(
    model.config, "max_position_embeddings", 2048
)  # Default to 2048 if not found
print(f"Model's context window: {context_length} tokens")

# Recommended max generation length (typically less than context length)
max_generation_length = min(1024, context_length - 10)  # Leave some room for input
print(f"Recommended max generation length: {max_generation_length} tokens")


def generate_response(prompt, max_new_tokens=300, temperature=0.7, top_p=0.9):
    """Improved generation function with better stopping criteria"""
    # Ensure prompt ends with newline if using instruction format
    if "### Instruction:" in prompt and not prompt.endswith("\n"):
        prompt += "\n"

    # Tokenize with return_tensors="pt" by default
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=context_length - 100
    ).to(device)

    # Generate with better stopping criteria
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        streamer=streamer,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Decode while removing special tokens and prompt
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_output[len(prompt) :]  # Remove prompt from output

    return response.strip()


# Example usage
prompt = "### Instruction: Do sky matching between 'cat1.fits' and 'cat2.fits'.\n### Response:"
print("\nGenerating response...")
response = generate_response(prompt, max_new_tokens=500, temperature=0.8, top_p=0.95)

print("\nFinal Response:")
print(response)
