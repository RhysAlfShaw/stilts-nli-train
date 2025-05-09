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


# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-1B"
OUTPUT_DIR = "./stilts-llm-finetuned"
TRAIN_FILE = "training_data.json"

# MAX_LENGTH = 512
BATCH_SIZE = 12
LEARNING_RATE = 2e-5
NUM_EPOCHS = 100
GRADIENT_ACCUMULATION_STEPS = 4

# Load access token
with open("access_token", "r") as f:
    access_token = f.read().strip()
os.environ["HF_TOKEN"] = access_token

# Check for GPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# set a max usage of GPU memory
if device.type == "cuda":
    desired_max = 30  # in GB
    max_menory = torch.cuda.get_device_properties(0).total_memory
    fraction = desired_max * 1024**3 / max_menory

    torch.cuda.set_per_process_memory_fraction(0.8)

    print("Set GPU memory usage to 50%")

# Create output directory

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the tokenizer and model

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # Use EOS token for padding

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

# Add special tokens for instruction formatting

special_tokens_dict = {
    "additional_special_tokens": ["### Instruction:", "### Response:"]
}
tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

# Load and prepare the dataset
print("Preparing dataset...")
with open(TRAIN_FILE, "r") as f:
    data = json.load(f)

# Format the data for instruction fine-tuning

template = "{% for message in messages %}{{'<|start_header_id|>' + message['role'] + '<|end_header_id|>\n' + message['content'] + '<|eot_id|>' + '\n'}}{% endfor %}"
formatted_data = []
for item in data:
    prompt = item["prompt"]
    response = item["response"]
    chat = [
        {
            "role": "system",
            "content": "You are a stilts command generator.",
        },
        {"role": "user", "content": item["prompt"]},
        {"role": "assistant", "content": item["response"]},
    ]
    formatted = tokenizer.apply_chat_template(
        chat, chat_template=template, tokenize=False
    )
    formatted_data.append({"text": formatted})

# Create dataset
dataset = Dataset.from_list(formatted_data)
print("Number of examples in dataset:", len(dataset))


# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
    )


tokenized_dataset = dataset.map(
    tokenize_function, batched=True, remove_columns=["text"]
)

# Debug: Check EOS tokens are properly included

print("\nVerifying EOS tokens in dataset:")
for i in range(min(2, len(tokenized_dataset))):  # Check first 2 examples
    print(f"\nExample {i}:")
    print("Input IDs:", tokenized_dataset[i]["input_ids"])
    print("Decoded:", tokenizer.decode(tokenized_dataset[i]["input_ids"]))

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


# # plot loss
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

# Plot training loss (blue)
plt.plot(steps, train_loss, label="Training Loss", color="blue")

# Plot eval loss (red, if available)
if eval_loss:
    plt.plot(
        eval_steps, eval_loss, label="Evaluation Loss", color="red", linestyle="--"
    )

plt.title("Training & Evaluation Loss Curve")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.yscale("log")  # Use log scale for better visibility
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig(f"{OUTPUT_DIR}/loss_curve.png")
