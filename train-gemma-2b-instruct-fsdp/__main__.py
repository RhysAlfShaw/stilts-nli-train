import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
import matplotlib.pyplot as plt

# --- Configuration ---
MODEL_NAME = "google/gemma-2b-it"
OUTPUT_DIR = "/scratch/Rhys/stilts_models/gemma-2b-it-finetuned-lora"
PLOT_OUTPUT_DIR = "."
TRAIN_FILE = "DATA/training_data.json"
BATCH_SIZE = 20
LEARNING_RATE = 5e-5
NUM_EPOCHS = 50
GRADIENT_ACCUMULATION_STEPS = 1
DTYPE = torch.bfloat16

print(f"Using model: {MODEL_NAME}")
print(f"Using output directory: {OUTPUT_DIR}")

# --- Setup ---
# Load access token
try:
    with open("access_token", "r") as f:
        access_token = f.read().strip()
    os.environ["HF_TOKEN"] = access_token
except FileNotFoundError:
    print(
        "WARNING: 'access_token' file not found. Ensure you are logged in via huggingface-cli."
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set a max usage of GPU memory (optional, but good practice)
if device.type == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.85)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Model and Tokenizer Loading ---
print("Loading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Add a padding token if it doesn't exist. Necessary for DataCollatorForLanguageModeling.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", torch_dtype=DTYPE
)

# --- PEFT LoRA Configuration ---
print("Applying PEFT LoRA configuration...")
lora_config = LoraConfig(
    r=8,  # Rank of the update matrices. Lower rank means fewer parameters to train.
    lora_alpha=16,  # Alpha scaling factor.
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],  # Apply LoRA to attention projections.
    lora_dropout=0.05,  # Dropout probability for LoRA layers.
    bias="none",
    task_type="CAUSAL_LM",
)

# Wrap the base model with the PEFT model
model = get_peft_model(model, lora_config)

# Print a summary of the trainable parameters
model.print_trainable_parameters()

# --- Dataset Preparation ---
print("Preparing dataset...")
with open(TRAIN_FILE, "r") as f:
    data = json.load(f)

# Format the data using the chat template
formatted_data = []
for item in data:
    chat = [
        {"role": "user", "content": item["prompt"]},
        {"role": "assistant", "content": item["response"]},
    ]
    # The template adds special tokens, including BOS and EOS
    formatted_data.append({"text": tokenizer.apply_chat_template(chat, tokenize=False)})

dataset = Dataset.from_list(formatted_data)


# Tokenize the dataset
def tokenize_function(examples):
    # We set padding to False here and handle it in the data collator
    return tokenizer(examples["text"], truncation=True, max_length=512, padding=False)


tokenized_dataset = dataset.map(
    tokenize_function, batched=True, remove_columns=["text"]
)

# Split into training and validation sets
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = tokenized_dataset["train"]
eval_dataset = tokenized_dataset["test"]

print(f"\nTraining dataset size: {len(train_dataset)}")
print(f"Evaluation dataset size: {len(eval_dataset)}")

# --- Training ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,  # Log more frequently to see progress
    save_strategy="epoch",
    eval_strategy="epoch",  # Evaluate at the end of each epoch
    load_best_model_at_end=True,  # Load the best model based on eval_loss
    metric_for_best_model="eval_loss",
    save_total_limit=1,
    report_to="tensorboard",
    push_to_hub=False,
    bf16=True,  # Use bfloat16 for training if available
)

# Data collator handles padding dynamically for each batch
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

print("\nStarting training...")
trainer.train()

# --- Save Model and Plot Results ---
print("\nSaving final LoRA adapter...")
# This saves only the trained LoRA adapter weights, not the full model.
trainer.save_model(f"{OUTPUT_DIR}/final_adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_adapter")

# Optional: Merge adapter and save full model
# print("\nMerging adapter and saving full model...")
# merged_model = model.merge_and_unload()
# merged_model.save_pretrained(f"{OUTPUT_DIR}/final_merged_model")
# tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_merged_model")

print("\nPlotting loss curve...")
if trainer.state.log_history:
    logs = trainer.state.log_history
    train_loss = [log["loss"] for log in logs if "loss" in log]
    train_steps = [log["step"] for log in logs if "loss" in log]
    eval_loss = [log["eval_loss"] for log in logs if "eval_loss" in log]
    eval_steps = [log["step"] for log in logs if "eval_loss" in log]

    plt.figure(figsize=(12, 7))
    plt.plot(train_steps, train_loss, label="Training Loss")
    if eval_loss:
        plt.plot(
            eval_steps, eval_loss, label="Evaluation Loss", marker="o", linestyle="--"
        )

    plt.title("Training & Evaluation Loss Curve")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plot_filename = f"{PLOT_OUTPUT_DIR}/{os.path.basename(OUTPUT_DIR)}_loss_curve.png"
    plt.savefig(plot_filename)
    print(f"Loss curve saved to {plot_filename}")
else:
    print("No log history found to plot loss curve.")
