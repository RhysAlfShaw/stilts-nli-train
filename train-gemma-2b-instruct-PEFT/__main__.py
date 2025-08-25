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

# base model.
MODEL_NAME = "google/gemma-2b-it"
OUTPUT_DIR = "/scratch/Rhys/stilts_models/gemma-2b-it-finetuned-peft"
PLOT_OUTPUT_DIR = "."
TRAIN_FILE = "DATA/training_data.json"
ADDITIONAL_TRAIN_FILES = [
    "DATA/training_data-tmatchn.json",
    "DATA/training_data-tpipe.json",
    "DATA/training_data-tpipe2.json",
    "DATA/training_data-tpipe3.json",
    "DATA/training_data-tmatch2.json",
    "DATA/training_data-descr.json",
    "DATA/training_data-descr-extr.json",
    "DATA/training_data-explanations.json",
    # "DATA/training_data-local-file.json",
    # "DATA/training_data-cmd-opts.json",
    # "DATA/training_data-basic-file-formats.json",
    "DATA/doc-examples-formatted.json",
]
EVAL_TEST_SPLIT = 0.1  # 30% for evaluation
BATCH_SIZE = 2  # good starting point.
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


# os.makedirs(OUTPUT_DIR, exist_ok=True)
# print("Loading tokenizer and model...")
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME, device_map="auto", torch_dtype=DTYPE
# )

from peft import LoraConfig, get_peft_model, TaskType
from transformers import BitsAndBytesConfig

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# --- 1. Configure 4-bit quantization (for memory efficiency) ---
# This will load the base model in 4-bit, drastically reducing memory usage.
# The small LoRA adapters will be trained in bfloat16.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # Use "nf4" for 4-bit NormalFloat
    bnb_4bit_compute_dtype=torch.bfloat16,  # Computation type
    bnb_4bit_use_double_quant=True,  # Optional, for extra memory savings
)

# --- 2. Load the base model with the quantization config ---
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    # quantization_config=bnb_config,
    device_map="auto",  # Automatically places parts of the model on the best device (GPU/CPU)
)
# Ensure tokenizer has a padding token
tokenizer.pad_token = tokenizer.eos_token

# --- 3. Define the LoRA Configuration ---
# This tells PEFT which parts of the model to adapt.
# For Gemma, targeting the attention projection layers is standard.
lora_config = LoraConfig(
    r=16,  # The rank of the LoRA matrices. Higher rank = more parameters, more expressivity. (8, 16, 32, 64 are common)
    lora_alpha=32,  # A scaling factor for the LoRA updates. A common rule of thumb is 2 * r.
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],  # Key layers to adapt in Gemma
    lora_dropout=0.05,  # Dropout for LoRA layers
    bias="none",  # Typically set to 'none' for LoRA
    task_type=TaskType.CAUSAL_LM,
)

# --- 4. Apply PEFT to the model ---
# This wraps the base model with the LoRA adapters.
print("\nApplying LoRA adapters to the model...")
model = get_peft_model(model, lora_config)

# --- 5. Print the trainable parameters ---
# You'll see that you are only training a tiny fraction of the total parameters!
model.print_trainable_parameters()

# ==============================================================================
# END OF NEW SCRIPT BLOCK
# ==============================================================================


# print("Preparing dataset...")
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
# Merge the LoRA layers with the base model
merged_model = model.merge_and_unload()

# Define a path for the final, merged model
final_model_path = f"{OUTPUT_DIR}/final_merged_model"

# Save the merged model and tokenizer
merged_model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)

print(f"Final merged model saved to {final_model_path}")

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
