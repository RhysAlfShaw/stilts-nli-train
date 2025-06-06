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
    GenerationConfig,
)
import matplotlib.pyplot as plt

MODEL_NAME = "distilgpt2"  # model size - 82M
OUTPUT_DIR = "./stilts-llm-finetuned-distilgpt2"
TRAIN_FILE = "training_data.json"
VALID_FILE = "testing_data.json"  # Added validation file


BATCH_SIZE = 48
LEARNING_RATE = 1e-4
NUM_EPOCHS = 500
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
    torch.cuda.set_per_process_memory_fraction(0.8)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the tokenizer and model
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # Use EOS token for padding

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

# Load the training data
print("Loading training data...")
with open(TRAIN_FILE, "r") as f:
    data = json.load(f)
    # Convert to Dataset object

custom_chat_template = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"  # Llama 3 specific: system prompt is optional but should be first if present
    "{{'<|begin_of_text|>' + '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>'}}"
    "{% else %}"
    "{{'<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>'}}"
    "{% endif %}"
    "{% if not loop.last %}"  # Add newline between turns but not after the very last EOT token.
    "\n"  # Or consider if the model expects a specific closing sequence or if apply_chat_template handles it
    "{% endif %}"
    "{% endfor %}"
)


def load_and_format_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    formatted_data = []
    for item in data:
        chat = [
            {
                "role": "system",
                "content": "You are a stilts command generator.",  # System prompt
            },
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["response"]},
        ]
        # Using the tokenizer's default chat template is recommended for Llama 3
        try:
            # We set add_generation_prompt=False because we provide the full chat history including the assistant's response.
            # The model should learn to complete the assistant's part.
            formatted_text = tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=False,  # Important for training
            )
        except Exception as e:
            print(
                f"Error applying default chat template: {e}. Falling back to custom template (ensure it's Llama 3 compatible)."
            )

            print(
                "Using the simplified custom template provided in the original script. Note: This might not be optimal for Llama 3."
            )
            template_from_script = "{% for message in messages %}{{'<|start_header_id|>' + message['role'] + '<|end_header_id|>\n' + message['content'] + '<|eot_id|>'}}{% if not loop.last %}{{'\n'}}{% endif %}{% endfor %}"
            formatted_text = tokenizer.apply_chat_template(
                chat,
                chat_template=template_from_script,  # Using the user's original template structure
                tokenize=False,
            )

        formatted_data.append({"text": formatted_text})
    return formatted_data


if os.path.exists(VALID_FILE):
    valid_formatted_data = load_and_format_data(VALID_FILE)
    dataset_valid = Dataset.from_list(valid_formatted_data)
    print(f"Number of examples in validation dataset: {len(dataset_valid)}")
else:
    print(f"Validation file {VALID_FILE} not found. Training without validation.")
    dataset_valid = None


template_llama = "{% for message in messages %}{{'<|start_header_id|>' + message['role'] + '<|end_header_id|>\n' + message['content'] + '<|eot_id|>' + '\n'}}{% endfor %}"
template_distilgpt2 = "{% for message in messages %}{{'<|start_header_id|>' + message['role'] + '<|end_header_id|>\n' + message['content'] + '<|eot_id|>' + '\n'}}{% endfor %}"
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
        chat, chat_template=template_distilgpt2, tokenize=False
    )
    formatted_data.append({"text": formatted})

# print example of training data
print("\nExample of training data:")
print("Formatted data:", formatted_data[0]["text"])
# exit()
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
    save_strategy="no",  # Save only at the end of training
    save_total_limit=1,
    report_to="tensorboard",
    push_to_hub=False,
    eval_strategy="epoch",
    metric_for_best_model="eval_loss",
)

# Create data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

if dataset_valid:
    tokenized_valid_dataset = dataset_valid.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
else:
    tokenized_valid_dataset = None

# Create trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,  # Using the same dataset for eval
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Train the model

# Train the model
print("\nStarting training...")
try:
    trainer.train()
except Exception as e:
    print(f"Error during training: {e}")

    raise

# Save the fine-tuned model and tokenizer
print("\nSaving final model...")
final_model_path = f"{OUTPUT_DIR}/final_model"
model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)


eos_token_ids = [tokenizer.eos_token_id]
if tokenizer.vocab.get("<|eot_id|>") is not None:  # Llama3 specific end of turn
    eos_token_ids.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))
if (
    tokenizer.vocab.get("<|end_of_text|>") is not None
):  # Llama3 specific end of sequence
    eos_token_ids.append(tokenizer.convert_tokens_to_ids("<|end_of_text|>"))

# Remove duplicates
eos_token_ids = sorted(list(set(eos_token_ids)))


generation_config = GenerationConfig(
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=eos_token_ids,  # Can be a list of EOS token IDs for Llama 3
    max_new_tokens=256,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
generation_config.save_pretrained(final_model_path)
print(f"Generation config saved to {final_model_path}")


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
    loss_curve_path = f"{OUTPUT_DIR}/loss_curve.png"
    plt.savefig(loss_curve_path)
    print(f"Loss curve saved to {loss_curve_path}")
else:
    print("No log history found to plot loss curve.")

print("\nFine-tuning complete.")
print(f"Model and tokenizer saved in {final_model_path}")
