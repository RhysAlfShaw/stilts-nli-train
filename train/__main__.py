import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

with open('access_token', 'r') as f:
    access_token = f.read().strip()
os.environ["HF_TOKEN"] = access_token
# Configuration
MODEL_NAME = 'meta-llama/Llama-3.2-1B' # Replace with your downloaded model fits on Sotiria GPU for traning.
OUTPUT_DIR = "./stilts-llm-finetuned"
TRAIN_FILE = "training_data.json"
MAX_LENGTH = 512
BATCH_SIZE = 6
LEARNING_RATE = 2e-5 
NUM_EPOCHS = 40
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
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto"
)

# Load and prepare the dataset
print("Preparing dataset...")
with open(TRAIN_FILE, "r") as f:
    data = json.load(f)

# Format the data for instruction fine-tuning
formatted_data = []
for item in data:
    prompt = item["prompt"]
    response = item["response"]
    # Create instruction format: ### Instruction: {prompt}\n### Response: {response}
    formatted_text = f"### Instruction: {prompt}\n### Response: {response}"
    formatted_data.append({"text": formatted_text})

# Create dataset
dataset = Dataset.from_list(formatted_data)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
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
    save_total_limit=2,
    report_to="tensorboard",
    push_to_hub=False,
)

# Create data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # We're doing causal language modeling, not masked
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# Train the model
print("Starting training...")
trainer.train()

# Save the fine-tuned model
print("Saving model...")
model.save_pretrained(f"{OUTPUT_DIR}/final_model")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")

print("Fine-tuning complete!")

# Add inference example
print("\nExample usage after fine-tuning:")
print("""
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned model
model_path = "./stilts-llm-finetuned/final_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Example inference
prompt = "### Instruction: How do I create a color-magnitude diagram from my star catalog?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    inputs["input_ids"],
    max_length=512,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
""")