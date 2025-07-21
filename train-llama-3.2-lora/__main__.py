#
# PEFT Pre-training Example: Llama 3.2 on STILTS Documentation
#
# This script demonstrates how to use Parameter-Efficient Fine-Tuning (PEFT),
# specifically Low-Rank Adaptation (LoRA), to continue the pre-training of
# the Llama-3.2-1B-Instruct model on a specific knowledge domain.
#
# We will use the 'unsloth' library for a ~2x faster and 60% less memory-intensive
# training experience, making this feasible on a standard Colab notebook GPU.
#
# The process involves:
# 1. Installing necessary libraries.
# 2. Downloading and parsing the STILTS documentation (sun256.html).
# 3. Loading the Llama 3.2 1B model with 4-bit quantization (QLoRA).
# 4. Configuring and applying LoRA adapters.
# 5. Training the model on the new text.
# 6. Showing a before-and-after comparison to verify the new knowledge.
#

# Step 1: Install and import necessary libraries
# We use unsloth for performance, transformers for model handling, peft for LoRA,
# and beautifulsoup4 for HTML parsing.

import torch
import requests
from bs4 import BeautifulSoup
from datasets import Dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer

# Step 2: Acquire and Prepare the Training Data
# We will download the single-page HTML documentation for STILTS and parse it.
print("=== Step 2: Acquiring and Preparing Data ===")

# URL to the STILTS single-page HTML documentation
STILTS_DOCS_URL = "https://www.star.bris.ac.uk/mbt/stilts/sun256/sun256.html"

# load .txt from local
datapath = "sun256.txt"
with open(datapath, "r", encoding="utf-8") as file:
    stilts_text = file.read()
print(f"Loaded STILTS documentation from {datapath}.")
data = {"text": [stilts_text]}
dataset = Dataset.from_dict(data)

print("Dataset created successfully.")


# Step 3: Load the Pre-trained Model and Tokenizer
# We use unsloth's FastLanguageModel to load the Llama 3.2 model with 4-bit
# quantization, which drastically reduces memory usage.
print("\n=== Step 3: Loading Llama 3.2 Model ===")

max_seq_length = 2048  # Choose any! We'll use 2048 to manage memory.
dtype = None  # None for auto-detection. Float16 for Tesla T4/V100, Bfloat16 for Ampere.
load_in_4bit = False  # Use 4-bit quantization to save memory.

# Load the model and tokenizer from Hugging Face
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    # max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Step 4: Configure PEFT (LoRA)
# We add LoRA adapters to the model. This is where unsloth's magic happens,
# preparing the model for efficient fine-tuning. Only these small adapters
# will be trained.
print("\n=== Step 4: Configuring PEFT (LoRA) ===")

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank. Suggested values: 8, 16, 32, 64, 128
    lora_alpha=32,  # LoRA alpha. Alpha = 2 * Rank is a common starting point.
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,  # Dropout probability
    bias="none",  # "none" or "all"
    use_gradient_checkpointing=True,
    random_state=42,
    use_rslora=False,  # Rank-Stabilized LoRA
    loftq_config=None,  # LoRA-Fine-Tuning Quantization
)
print("PEFT model configured successfully.")
model.print_trainable_parameters()


# Step 5: Define Training Arguments and Start Training
print("\n=== Step 5: Starting Training ===")

# Define the trainer using TRL's SFTTrainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",  # The column in our dataset with the text
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=100,  # Set a small number of steps for this demo. Increase for better results.
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        # optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir="outputs",
    ),
)

# Start the training process
training_stats = trainer.train()

print("\nTraining complete!")
print("Stats:", training_stats)


# Step 6: Inference and Verification
# Let's see if the model learned about STILTS! We will ask a specific question
# that it likely couldn't answer before, but should be able to answer now.
print("\n=== Step 6: Inference and Verification ===")

# Unsloth-style prompt formatting
prompt_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# We'll use the same prompt for both the base and fine-tuned model
instruction_text = "Answer the following question about the STILTS command-line tool."
input_text = (
    "What is the command to crossmatch two tables on sky position using STILTS?"
)

# --- Inference with the Fine-Tuned Model ---
# The `trainer.model` is our PEFT-adapted model
inputs = tokenizer(
    [
        prompt_template.format(
            instruction_text,
            input_text,
            "",  # Response is empty
        )
    ],
    return_tensors="pt",
).to("cuda")

outputs = trainer.model.generate(**inputs, max_new_tokens=64, use_cache=True)
fine_tuned_response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

print("\n--- AFTER Fine-Tuning ---")
print(f"Question: {input_text}")
print("Response:")
print(fine_tuned_response.split("### Response:")[1].strip())


# --- For comparison, let's load the original base model and ask it ---
# NOTE: This requires reloading the model, which uses VRAM. This part might
# fail on very memory-constrained GPUs.
try:
    print("\nLoading original base model for comparison...")
    base_model, _ = FastLanguageModel.from_pretrained(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        # max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    base_model_inputs = tokenizer(
        [
            prompt_template.format(
                instruction_text,
                input_text,
                "",  # Response is empty
            )
        ],
        return_tensors="pt",
    ).to("cuda")

    base_outputs = base_model.generate(
        **base_model_inputs, max_new_tokens=64, use_cache=True
    )
    base_model_response = tokenizer.batch_decode(
        base_outputs, skip_special_tokens=True
    )[0]

    print("\n--- BEFORE Fine-Tuning (Original Model) ---")
    print(f"Question: {input_text}")
    print("Response:")
    print(base_model_response.split("### Response:")[1].strip())

    print(
        "\nCompare the two responses. The fine-tuned model should provide a more accurate and specific answer based on the documentation."
    )

except Exception as e:
    print(
        f"\nCould not load base model for comparison due to an error (likely VRAM): {e}"
    )
