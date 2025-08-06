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

MODEL_NAME = "google/gemma-2b-it"
OUTPUT_DIR = "/scratch/Rhys/stilts_models/gemma-2b-it-finetuned"
PLOT_OUTPUT_DIR = "."
TRAIN_FILE = "DATA/training_data.json"
ADDITIONAL_TRAIN_FILES = [
    "DATA/training_data-tmatchn.json",
    "DATA/training_data-tpip.json",
    "DATA/training_data-tmatch2.json",
    "DATA/training_data-descr.json",
    "DATA/training_data-local-file.json",
    "DATA/training_data-cmd-opts.json",
    "DATA/training_data-basic-file-formats.json",
    "DATA/doc-examples-formatted.json",
]
EVAL_TEST_SPLIT = 0.3  # 30% for evaluation
BATCH_SIZE = 1  # good starting point.
LEARNING_RATE = 5e-5  # 5e-5 is a common learning rate for fine-tuning large models
NUM_EPOCHS = 5  # 3 epochs is a good starting point for fine-tuning
GRADIENT_ACCUMULATION_STEPS = 1
DTYPE = torch.bfloat16  # important to keep memory usage low for large models.

print(f"Using model: {MODEL_NAME}")
print(f"Using output directory: {OUTPUT_DIR}")
# Load access token

with open("access_token", "r") as f:
    access_token = f.read().strip()
os.environ["HF_TOKEN"] = access_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# set a max usage of GPU memory
if device.type == "cuda":
    # set to 80% of total memory
    torch.cuda.set_per_process_memory_fraction(0.60)


os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", torch_dtype=DTYPE
)

print("Preparing dataset...")
with open(TRAIN_FILE, "r") as f:
    data = json.load(f)

# Load additional training data if specified
if ADDITIONAL_TRAIN_FILES:
    for additional_file in ADDITIONAL_TRAIN_FILES:
        with open(additional_file, "r") as f:
            additional_data = json.load(f)
            data.extend(additional_data)


formatted_data = []
for item in data:
    chat = [
        {"role": "user", "content": item["prompt"]},
        {"role": "assistant", "content": item["response"]},
    ]
    formatted_data.append({"text": tokenizer.apply_chat_template(chat, tokenize=False)})

# formatted_data = tokenizer.apply_chat_template(data, tokenize=False)
dataset = Dataset.from_list(formatted_data)


# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
    )


tokenized_dataset = dataset.map(
    tokenize_function, batched=True, remove_columns=["text"]
)
# split tokenized_dataset into train and validation sets
tokenized_dataset = tokenized_dataset.train_test_split(
    test_size=EVAL_TEST_SPLIT, seed=42
)

test_dataset = tokenized_dataset["test"]
tokenized_dataset = tokenized_dataset["train"]

print(f"\nTokenized dataset size: {len(tokenized_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# print("\nVerifying EOS tokens in dataset:")
# for i in range(min(2, len(tokenized_dataset))):  # Check first 2 examples
#     print(f"\nExample {i}:")
#     print("Input IDs:", tokenized_dataset[i]["input_ids"])
#     print("Decoded:", tokenizer.decode(tokenized_dataset[i]["input_ids"]))

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
    metric_for_best_model="eval_loss",
    eval_strategy="epoch",
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
)


print("\nStarting training...")
trainer.train()

print("\nSaving model...")
model.save_pretrained(f"{OUTPUT_DIR}/final_model")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")

# # plot loss
import matplotlib.pyplot as plt

# Plot loss
print("\nPlotting loss curve...")
if trainer.state.log_history:
    logs = trainer.state.log_history
    train_loss = [log["loss"] for log in logs if "loss" in log]
    train_steps = [log["step"] for log in logs if "loss" in log]
    eval_loss = [log["eval_loss"] for log in logs if "eval_loss" in log]
    eval_steps = [
        log["step"] for log in logs if "eval_loss" in log
    ]  # eval_loss is logged at same step as a training log point

    plt.figure(figsize=(12, 7))
    plt.plot(train_steps, train_loss, label="Training Loss", color="blue", alpha=0.7)
    print(eval_loss)
    if eval_loss:
        # Align eval steps with training steps for better plotting if needed,
        # or use eval_steps directly if they are logged with 'step' key correctly.
        # Trainer logs eval_loss with the global step at which evaluation happened.
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
    plt.savefig(f"{PLOT_OUTPUT_DIR}/loss_curve.png")
else:
    print("No log history found to plot loss curve.")
# Save the plot

# now run the hf_to_gguf.py script to convert the model to GGUF format
print("\nConverting model to GGUF format...")
import subprocess

# this is not working atm.?
cmd = f"python hf_to_gguf.py --model_path {OUTPUT_DIR}/final_model --output_path {OUTPUT_DIR}/final_model.gguf"
subprocess.run(cmd, shell=True, check=True)
print("\nTraining complete and model saved in GGUF format.")
