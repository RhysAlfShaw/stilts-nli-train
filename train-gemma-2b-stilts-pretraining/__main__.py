import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import matplotlib.pyplot as plt
import subprocess

# --- Configuration ---

# 1. Model and Directories
# For pre-training, we use the base model, not the instruction-tuned (-it) version.
MODEL_NAME = "google/gemma-2b"
OUTPUT_DIR = "/scratch/Rhys/stilts_models/gemma-2b-pretrained-sun256-no-quant"  # Changed output dir
PLOT_OUTPUT_DIR = "."
TRAIN_FILE = "DATA/sun256.txt"

# 2. Training Hyperparameters
BATCH_SIZE = 2  # Reduced batch size due to higher memory usage without quantization
LEARNING_RATE = 1e-5  # Increased learning rate for pre-training
NUM_EPOCHS = 1
GRADIENT_ACCUMULATION_STEPS = 8  # Increased to compensate for the smaller batch size
# 32 bit precision for better training stability during pre-training
DTYPE = torch.bfloat16
BLOCK_SIZE = 1024  # Context length

print(f"Using model: {MODEL_NAME}")
print(f"Using output directory: {OUTPUT_DIR}")

# --- Setup ---

# Load Hugging Face access token
try:
    with open("access_token", "r") as f:
        access_token = f.read().strip()
    os.environ["HF_TOKEN"] = access_token
except FileNotFoundError:
    print(
        "WARNING: 'access_token' file not found. Make sure you are logged in to Hugging Face CLI."
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Model and Tokenizer Loading ---

print("Loading tokenizer and model (no quantization)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Set a padding token if the tokenizer doesn't have one
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,  # Set the desired precision directly
    device_map="auto",  # Automatically distribute the model across available GPUs
    attn_implementation="eager",  # Use Flash Attention 2 if available
)


# --- Dataset Preparation for Pre-training ---

print("Preparing dataset for pre-training...")

# Load dataset from the text file
raw_dataset = load_dataset("text", data_files={"train": TRAIN_FILE})


# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"])


tokenized_dataset = raw_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    desc="Running tokenizer on dataset",
)


# Group texts into blocks of size BLOCK_SIZE
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder.
    if total_length >= BLOCK_SIZE:
        total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
    # Split by chunks of BLOCK_SIZE.
    result = {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


lm_dataset = tokenized_dataset.map(
    group_texts,
    batched=True,
    desc=f"Grouping texts into chunks of {BLOCK_SIZE}",
)

# Split into training and evaluation sets
# split_dataset = lm_dataset["train"].train_test_split(test_size=0.1, seed=42)
# train_dataset = split_dataset["train"]
# eval_dataset = split_dataset["test"]
train_dataset = lm_dataset["train"]
eval_dataset = lm_dataset["train"]


print(f"\nProcessed training dataset size: {len(train_dataset)}")


# --- Trainer Setup ---

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=1,
    report_to="tensorboard",
    push_to_hub=False,
    # bf16/fp16 flags are set based on DTYPE
    bf16=True if DTYPE == torch.bfloat16 else False,
    fp16=True if DTYPE == torch.float16 else False,
)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)


# --- Training ---

print("\nStarting training...")
trainer.train()

# --- Saving ---

print("\nSaving final model...")
final_model_path = f"{OUTPUT_DIR}/final_model"
model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)


# --- Plotting Loss ---

print("\nPlotting loss curve...")
if trainer.state.log_history:
    logs = trainer.state.log_history
    train_loss = [log["loss"] for log in logs if "loss" in log]
    train_steps = [log["step"] for log in logs if "loss" in log]
    eval_loss = [log["eval_loss"] for log in logs if "eval_loss" in log]
    eval_steps = [log["step"] for log in logs if "eval_loss" in log]

    plt.figure(figsize=(12, 7))
    plt.plot(train_steps, train_loss, label="Training Loss", color="blue", alpha=0.7)

    if eval_loss:
        plt.plot(
            eval_steps,
            eval_loss,
            label="Evaluation Loss",
            color="red",
            marker="o",
            linestyle="--",
        )

    plt.title("Training & Evaluation Loss Curve")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig(f"{PLOT_OUTPUT_DIR}/loss_curve_pretrained_no_quant.png")
    print(f"Loss curve saved to {PLOT_OUTPUT_DIR}/loss_curve_pretrained_no_quant.png")
else:
    print("No log history found to plot loss curve.")
