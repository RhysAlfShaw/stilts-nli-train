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
    GenerationConfig,  # Added for saving generation config
)
import matplotlib.pyplot as plt

# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_DIR = "./stilts-llm-finetuned"
TRAIN_FILE = "training_data.json"
VALID_FILE = "testing_data.json"  # Added validation file

MAX_LENGTH = 1024  # Max sequence length for tokenizer (adjust as needed)
BATCH_SIZE = 8  # Adjusted for potentially larger MAX_LENGTH and model size
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5  # Reduced from 100 to a more reasonable number for fine-tuning
GRADIENT_ACCUMULATION_STEPS = (
    4  # Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
)

# Load access token
try:
    with open("access_token", "r") as f:
        access_token = f.read().strip()
    os.environ["HF_TOKEN"] = access_token
except FileNotFoundError:
    print(
        "Access token file 'access_token' not found. Proceeding without it if model is public or cached."
    )
    access_token = None

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set a max usage of GPU memory (optional, and an alternative is PYTORCH_CUDA_ALLOC_CONF)
if device.type == "cuda":
    # The following lines attempt to set a memory fraction.
    # Note: `torch.cuda.set_per_process_memory_fraction` takes a value between 0.0 and 1.0.
    # The `desired_max` and `fraction` calculation below is one way to derive this,
    # but it's not used in the `set_per_process_memory_fraction` call directly.
    # `device_map="auto"` also helps manage memory.
    # Consider using environment variable `PYTORCH_CUDA_ALLOC_CONF` for more robust control.
    try:
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Total GPU Memory: {total_memory_gb:.2f} GB")
        # Example: Set to use 80% of GPU memory. Adjust as needed.
        torch.cuda.set_per_process_memory_fraction(0.8)
        print("Attempted to set GPU memory usage to 80% of the first GPU.")
        # The `desired_max` logic was not previously connected, kept for reference:
        # desired_max_gb = 30  # Desired max in GB
        # if total_memory_gb > 0:
        #     calculated_fraction = min(1.0, desired_max_gb / total_memory_gb)
        #     # torch.cuda.set_per_process_memory_fraction(calculated_fraction, device=0) # Example usage
    except Exception as e:
        print(f"Could not set GPU memory fraction: {e}")


# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the tokenizer and model
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=access_token)

# It's crucial for Causal LMs that a pad token is defined.
# Llama 3 tokenizer might not have a pad token by default. Using EOS token for padding.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Pad token set to EOS token: {tokenizer.eos_token}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",  # Automatically distributes model layers across available devices (GPU, CPU)
    torch_dtype=torch.bfloat16,  # Use bfloat16 for faster training and reduced memory if supported
    token=access_token,
)

# Resize token embeddings if new tokens were added (not the case here as special_tokens_dict is commented)
# model.resize_token_embeddings(len(tokenizer)) # Only needed if tokenizer vocabulary was changed

# Load and prepare the datasets
print("Preparing datasets...")

# --- Chat Template ---
# This is the Llama 3 specific chat template structure.
# It's HIGHLY recommended to use the tokenizer's default chat template if possible,
# as it's optimized for the model.
# To use the default:
# formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
# The custom template below should be an EXACT match if you intend to override.
# Verify newlines, spaces, and special tokens like <|begin_of_text|>.
# Llama 3's apply_chat_template usually adds <|begin_of_text|> by default.
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
    # Ensure the final assistant message also ends correctly (typically with its <|eot_id|>)
    # The trainer will handle labels, model learns to predict the whole sequence including assistant response.
)
# After discussion, it's usually better to use the tokenizer's default.
# If you must use a custom one, ensure it's correct.
# For Llama 3, `add_generation_prompt=False` for training data is typical as you provide the full conversation.
# `add_generation_prompt=True` is for inference when you want the model to start the assistant's turn.


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
            # Fallback or strict use of custom template if required:
            # formatted_text = tokenizer.apply_chat_template(
            #     chat,
            #     chat_template=custom_chat_template, # Make sure this is Llama 3 compatible!
            #     tokenize=False
            # )
            # Forcing a custom template like this can be risky if not perfectly aligned with model pre-training.
            # The example above is a conceptual custom_chat_template string.
            # The original script's template was simpler:
            # template_from_script = "{% for message in messages %}{{'<|start_header_id|>' + message['role'] + '<|end_header_id|>\n' + message['content'] + '<|eot_id|>' + '\n'}}{% endfor %}"
            # This simpler template might miss nuances like specific newline counts or BOS token.

            # Using the original script's simpler template as per user's initial code, but with a warning:
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


# Load datasets
train_formatted_data = load_and_format_data(TRAIN_FILE)
dataset_train = Dataset.from_list(train_formatted_data)
print(f"Number of examples in training dataset: {len(dataset_train)}")

if os.path.exists(VALID_FILE):
    valid_formatted_data = load_and_format_data(VALID_FILE)
    dataset_valid = Dataset.from_list(valid_formatted_data)
    print(f"Number of examples in validation dataset: {len(dataset_valid)}")
else:
    print(f"Validation file {VALID_FILE} not found. Training without validation.")
    dataset_valid = None


# Tokenize the datasets
def tokenize_function(examples):
    tokenized_output = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        # padding="max_length", # Consider if needed; collator usually handles padding
    )
    return tokenized_output


print("\nTokenizing datasets...")
tokenized_train_dataset = dataset_train.map(
    tokenize_function, batched=True, remove_columns=["text"]
)

if dataset_valid:
    tokenized_valid_dataset = dataset_valid.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
else:
    tokenized_valid_dataset = None

# Debug: Check tokens
print("\nVerifying tokenization (first example of training data):")
if len(tokenized_train_dataset) > 0:
    example_train_ids = tokenized_train_dataset[0]["input_ids"]
    # print("Input IDs (train):", example_train_ids)
    print("Decoded (train):", tokenizer.decode(example_train_ids))
    # Check for EOS token (e.g., Llama 3's <|eot_id|> which is 128009, <|end_of_text|> is 128001)
    # print(f"EOS token ID: {tokenizer.eos_token_id}, <|eot_id|>: {tokenizer.convert_tokens_to_ids('<|eot_id|>') if '<|eot_id|>' in tokenizer.vocab else 'N/A'}")
    # if tokenizer.eos_token_id in example_train_ids:
    #     print("EOS token found in the first training example.")
    # else:
    #     print("EOS token NOT found in the first training example. This might be an issue if not intended.")

else:
    print("Training dataset is empty.")


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
    logging_steps=1,  # Log every 10 steps
    # save_strategy="epoch",
    save_total_limit=1,  # Save last 2 checkpoints
    report_to="tensorboard",
    push_to_hub=False,
    eval_strategy="epoch",
    #     "epoch" if tokenized_valid_dataset else "no"
    # ),  # Evaluate at the end of each epoch if validation set exists
    # load_best_model_at_end=(
    #     True if tokenized_valid_dataset else False
    # ),  # Load the best model at the end of training
    metric_for_best_model="eval_loss",
    bf16=torch.cuda.is_available()
    and torch.cuda.get_device_capability(0)[0]
    >= 8,  # Use bfloat16 if available (Ampere GPUs and newer)
    # fp16=not (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8), # Use fp16 if bf16 not available
)

# Create data collator
# DataCollatorForLanguageModeling is used for Causal LM. It handles padding.
# Labels are created automatically by shifting input_ids.
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # This is for Causal LM, not Masked LM
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_valid_dataset,  # Pass the validation dataset
    data_collator=data_collator,
    tokenizer=tokenizer,  # Passing tokenizer to save it along with the model
)

# Train the model
print("\nStarting training...")
try:
    trainer.train()
except Exception as e:
    print(f"Error during training: {e}")
    # You might want to save a checkpoint here if an error occurs mid-training
    # model.save_pretrained(f"{OUTPUT_DIR}/interrupted_model")
    # tokenizer.save_pretrained(f"{OUTPUT_DIR}/interrupted_model")
    raise

# Save the fine-tuned model and tokenizer
print("\nSaving final model...")
final_model_path = f"{OUTPUT_DIR}/final_model"
model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)

# Save generation config to ensure proper stopping during inference
# Llama 3 uses specific EOS tokens. The tokenizer should handle this,
# but explicit generation config can be useful.
# Common Llama 3 EOS tokens: <|eot_id|>, <|end_of_text|>
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
    # Example generation parameters (adjust as needed for stilts command generation)
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
    loss_curve_path = f"{OUTPUT_DIR}/loss_curve.png"
    plt.savefig(loss_curve_path)
    print(f"Loss curve saved to {loss_curve_path}")
else:
    print("No log history found to plot loss curve.")

print("\nFine-tuning complete.")
print(f"Model and tokenizer saved in {final_model_path}")
