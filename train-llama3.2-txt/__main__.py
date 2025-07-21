#
# PEFT Pre-training Example: Llama 3.2 on STILTS Documentation (Improved Version)
#
# This script demonstrates how to use Parameter-Efficient Fine-Tuning (PEFT),
# specifically Low-Rank Adaptation (LoRA), to continue the pre-training of
# the Llama-3.2-1B-Instruct model on a specific knowledge domain.
#
# KEY IMPROVEMENTS:
# 1.  Data Chunking: The raw text from the documentation is now split into
#     smaller paragraphs. This creates a more effective dataset for training.
# 2.  Increased Training Steps: Training steps have been increased from 60 to 200
#     to give the model enough time to learn the new information.
# 3.  Efficient Training: We use 'packing' to combine small text chunks, which
#     significantly speeds up the training process.
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
print("=== Step 2: Acquiring and Preparing Data ===")

# URL to the STILTS single-page HTML documentation
STILTS_DOCS_URL = "https://www.star.bris.ac.uk/mbt/stilts/sun256/sun256.html"

try:
    response = requests.get(STILTS_DOCS_URL)
    response.raise_for_status()  # Raise an exception for bad status codes

    # Use BeautifulSoup to parse the HTML and extract text
    soup = BeautifulSoup(response.content, "html.parser")
    full_text = soup.get_text(separator="\n", strip=True)

    # *** IMPROVEMENT: Chunk the text into smaller paragraphs ***
    # This creates more meaningful, bite-sized training examples for the model.
    # We split by double newlines, which typically separate paragraphs.
    text_chunks = [p.strip() for p in full_text.split("\n\n") if p.strip()]

    print(f"Successfully downloaded and parsed documentation.")
    print(f"Created {len(text_chunks)} text chunks for training.")

    # Create a Hugging Face Dataset object. The SFTTrainer expects a 'text' column.
    data = {"text": text_chunks}
    dataset = Dataset.from_dict(data)

    print("Dataset created successfully.")

except requests.exceptions.RequestException as e:
    print(f"Error downloading documentation: {e}")
    # As a fallback, use a small snippet of text if download fails
    text_chunks = [
        "STILTS is a set of command-line tools for processing tabular data.",
        "The command for crossmatching two tables on sky position is tskymatch2.",
        "The tpipe command performs pipeline processing on a table.",
        "votlint is a program which can check a VOTable document.",
    ]
    data = {"text": text_chunks}
    dataset = Dataset.from_dict(data)
    print("Using fallback dataset.")


# Step 3: Load the Pre-trained Model and Tokenizer
print("\n=== Step 3: Loading Llama 3.2 Model ===")

# max_seq_length = 2048  # Choose any! We'll use 2048 to manage memory.
dtype = None  # None for auto-detection. Float16 for Tesla T4/V100, Bfloat16 for Ampere.
load_in_4bit = True  # Use 4-bit quantization to save memory.

# Load the model and tokenizer from Hugging Face
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    # max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Step 4: Configure PEFT (LoRA)
print("\n=== Step 4: Configuring PEFT (LoRA) ===")

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank. Suggested values: 8, 16, 32, 64
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
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=42,
)
print("PEFT model configured successfully.")
model.print_trainable_parameters()


# Step 5: Define Training Arguments and Start Training
print("\n=== Step 5: Starting Training ===")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    # max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=True,  # *** IMPROVEMENT: Pack multiple short sequences for efficiency ***
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        # *** IMPROVEMENT: More training steps for better learning ***
        max_steps=30,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
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
print("\n=== Step 6: Inference and Verification ===")

# Unsloth-style prompt formatting for instruction-following
prompt_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

instruction_text = "Answer the following question about the STILTS command-line tool based on its documentation."
input_text = "What is the specific STILTS command for crossmatching two tables based on their sky positions?"

# --- Inference with the Fine-Tuned Model ---
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
# A bit of string manipulation to extract only the response part
print(fine_tuned_response.split("### Response:")[1].strip())


# --- For comparison, let's load the original base model and ask it ---
try:
    print("\nLoading original base model for comparison...")
    FastLanguageModel.for_inference(model)  # Prepare for inference

    base_model, _ = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
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
        "\nCompare the two responses. The fine-tuned model should now provide a much more accurate answer."
    )

except Exception as e:
    print(
        f"\nCould not load base model for comparison due to an error (likely VRAM): {e}"
    )


# now train on a fine-tuned dataset of prompt and response pairs
# Step 7: Save the Fine-Tuned Model
