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
    BitsAndBytesConfig,
)

from peft import LoraConfig, get_peft_model, TaskType

# set per process memory fraction with pytorch

torch.cuda.set_per_process_memory_fraction(0.1, 0)

# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-1B"
OUTPUT_DIR = "./stilts-llm-finetuned-lora"
TRAIN_FILE = "training_data.json"
MAX_LENGTH = 512
BATCH_SIZE = 24
LEARNING_RATE = 2e-5
NUM_EPOCHS = 1000
GRADIENT_ACCUMULATION_STEPS = 4

# LoRA Configuration.
LORA_R = 8  # Rank
LORA_ALPHA = 32  # Alpha parameter (scaling factor)
LORA_DROPOUT = 0.1  # Dropout probability

# Load access token
with open("access_token", "r") as f:
    access_token = f.read().strip()
os.environ["HF_TOKEN"] = access_token

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the tokenizer and model
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # Use EOS token for padding

# Optional: Quantization config (uncomment if you want 4-bit quantization)
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=False,
# )

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    # quantization_config=bnb_config,  # Uncomment if using quantization
    torch_dtype=torch.float16,
)

# Add special tokens for instruction formatting
special_tokens_dict = {
    "additional_special_tokens": ["### Instruction:", "### Response:"]
}
tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

# Define LoRA config - targeting output layers
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=[
        "q_proj",  # Query projection
        "v_proj",  # Value projection
        "o_proj",  # Output projection
        "lm_head",  # Language modeling head
    ],
    bias="none",
)

# Convert model to PEFT model with LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Print the number of trainable parameters

# Load and prepare the dataset
print("Preparing dataset...")
with open(TRAIN_FILE, "r") as f:
    data = json.load(f)

# Format the data for instruction fine-tuning
formatted_data = []
for item in data:
    prompt = item["prompt"]
    response = item["response"]
    # Create instruction format (let tokenizer handle EOS)
    formatted_text = f"### Instruction: {prompt}\n### Response: {response}"
    formatted_data.append({"text": formatted_text})

# Create dataset
dataset = Dataset.from_list(formatted_data)
print("Number of examples in dataset:", len(dataset))


# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
        add_special_tokens=True,  # Let this handle EOS automatically
    )


tokenized_dataset = dataset.map(
    tokenize_function, batched=True, remove_columns=["text"]
)

# Define training arguments
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
    logging_steps=1,
    save_strategy="epoch",
    save_total_limit=1,
    report_to="tensorboard",
    push_to_hub=False,
    fp16=True,  # Enable mixed precision training
)

# Create data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Train the model
print("\nStarting training...")
trainer.train()

# Save the fine-tuned model
print("\nSaving model...")
model.save_pretrained(f"{OUTPUT_DIR}/final_model")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")

# Save generation config to ensure proper stopping during inference
generation_config = {
    "eos_token_id": tokenizer.eos_token_id,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": MAX_LENGTH,
}
with open(f"{OUTPUT_DIR}/final_model/generation_config.json", "w") as f:
    json.dump(generation_config, f)

print("\nFine-tuning complete!")

# Plot loss
import matplotlib.pyplot as plt

# Get all training logs
logs = trainer.state.log_history

# Extract training loss (logged every `logging_steps`)
train_loss = [log["loss"] for log in logs if "loss" in log]
steps = [log["step"] for log in logs if "loss" in log]

# Extract eval loss (if evaluation was done)
eval_loss = [log["eval_loss"] for log in logs if "eval_loss" in log]
eval_steps = [log["step"] for log in logs if "eval_loss" in log]

plt.figure(figsize=(10, 6))
plt.plot(steps, train_loss, label="Training Loss", color="blue")
if eval_loss:
    plt.plot(
        eval_steps, eval_loss, label="Evaluation Loss", color="red", linestyle="--"
    )
plt.title("Training & Evaluation Loss Curve")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.xscale("log")
plt.legend()
plt.grid(True)
plt.savefig(f"{OUTPUT_DIR}/loss_curve.png")
