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

with open("access_token", "r") as f:
    access_token = f.read().strip()
os.environ["HF_TOKEN"] = access_token

# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-1B"
OUTPUT_DIR = "./stilts-llm-finetuned"
TRAIN_FILE = "training_data.json"
MAX_LENGTH = 512
BATCH_SIZE = 6
LEARNING_RATE = 2e-5
NUM_EPOCHS = 60
GRADIENT_ACCUMULATION_STEPS = 4

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the tokenizer and model
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

# Important: Add this to ensure the model learns to use EOS token properly
tokenizer.add_eos_token = True

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

# Load and prepare the dataset
print("Preparing dataset...")
with open(TRAIN_FILE, "r") as f:
    data = json.load(f)

# Format the data for instruction fine-tuning
formatted_data = []
for item in data:
    prompt = item["prompt"]
    response = item["response"]
    # Create instruction format and EXPLICITLY add EOS token
    formatted_text = (
        f"### Instruction: {prompt}\n### Response: {response}{tokenizer.eos_token}"
    )
    formatted_data.append({"text": formatted_text})

# Create dataset
dataset = Dataset.from_list(formatted_data)
print("Number of examples in dataset:", len(dataset))


# Tokenize the dataset
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
        add_special_tokens=True,
    )
    # Ensure we have EOS tokens where we expect them
    for i in range(len(tokenized["input_ids"])):
        # If the sequence isn't full, add EOS token at the end
        if (
            tokenized["input_ids"][i][-1] != tokenizer.eos_token_id
            and torch.sum(tokenized["attention_mask"][i]) < MAX_LENGTH
        ):
            # Find the first padding position
            pad_pos = torch.argmin(tokenized["attention_mask"][i]).item()
            tokenized["input_ids"][i][pad_pos] = tokenizer.eos_token_id
    return tokenized


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
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=1,
    report_to="tensorboard",
    push_to_hub=False,
)

# Create data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    # This ensures the loss isn't computed on padding tokens
    pad_to_multiple_of=8 if tokenizer.pad_token_id is not None else None,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Train the model
print("Starting training...")
trainer.train()

# Save the fine-tuned model
print("Saving model...")
model.save_pretrained(f"{OUTPUT_DIR}/final_model")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")

print("Fine-tuning complete!")
