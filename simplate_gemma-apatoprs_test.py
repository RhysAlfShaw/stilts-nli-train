import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --- Configuration ---
# The original model you fine-tuned
BASE_MODEL_PATH = "google/gemma-2b-it"
# The path to your saved LoRA adapter weights
ADAPTER_PATH = "/scratch/Rhys/stilts_models/gemma-2b-it-finetuned-lora/final_adapter"
dtype = torch.bfloat16

# --- Model Loading ---

# 1. Load the tokenizer from the base model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

# 2. Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    device_map="auto",
    torch_dtype=dtype,
)

# 3. Load the PEFT model by applying the adapter to the base model
print(f"Loading LoRA adapter from {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
print("LoRA adapter loaded successfully.")


# --- Inference ---
Prompts = [
    "How can I change a fits file (testing.fits) to csv format using STILTS?",
    "I want to do sky cross-matching between catalogues DRUID.fits and pybdsf.fits using their RA and Dec columns and with a 6 arcsecond seperation.",
    "What is STILTS?",
]

for prompt in Prompts:
    print("#######")
    print(f"Prompt: {prompt}")

    # Create the chat message format
    chat = [
        {"role": "user", "content": prompt},
    ]
    # Apply the chat template
    prompt_template = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )

    # Tokenize the formatted prompt
    inputs = tokenizer(
        prompt_template,
        return_tensors="pt",
    ).to("cuda")

    # Generate the response
    outputs = model.generate(
        **inputs, max_new_tokens=150, do_sample=True, temperature=0.7
    )

    # Decode only the newly generated tokens
    response_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True
    )

    print("Response:")
    print(response_text)
    print("#######\n")
