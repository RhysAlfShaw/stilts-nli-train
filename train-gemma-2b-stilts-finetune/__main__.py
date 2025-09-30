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
import argparse

# expect the following args

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--model_path",
    type=str,
    default="/scratch/Rhys/stilts_models/gemma-2b-pretrained-sun256-no-quant/final_model",
)

argparser.add_argument(
    "--output_dir",
    type=str,
    default="/scratch/Rhys/stilts_models/gemma-2b-finetuned",
)

argparser.add_argument(
    "--plot_output_dir",
    type=str,
    default=".",
)

argparser.add_argument(
    "--train_file_dir",
    type=str,
    default="DATA",
)

argparser.add_argument(
    "--mem_red",
    type=bool,
    default=True,
)

argparser.add_argument(
    "--batch_size",
    type=int,
    default=2,
)

argparser.add_argument(
    "--gguf-consersion",
    type=bool,
    default=True,
)

TRAINING_DATA_DIR = argparser.parse_args().train_file_dir
OUTPUT_DIR = argparser.parse_args().output_dir
MODEL_NAME = argparser.parse_args().model_path
PLOT_OUTPUT_DIR = argparser.parse_args().plot_output_dir
limit_mem = argparser.parse_args().mem_red

TRAIN_FILE = f"{TRAINING_DATA_DIR}/training_data.json"
ADDITIONAL_TRAIN_FILES = [
    "training_data-tpipe.json",
    "training_data-tpipe2.json",
    "training_data-tpipe3.json",
    "training_data-tpipe4.json",
    "training_data-tmatch2.json",
    "training_data-tmatchn.json",
    "training_data-tmatchn2.json",
    "training_data-descr.json",
    "training_data-descr-extr.json",
    "training_data-explanations.json",
    "doc-examples-formatted.json",
    "training_data_tcat-claude.json",
    "training_data_tcat-gpt-oss.json",
    "training_data-tcopy.json",
    "training_data-tpipefunc-artith.json",
    "training_data-tpipefunc-array.json",
    "training_data-tpipefunc-bits.json",
    "training_data-tpipefunc-conversion.json",
    "training_data-tpipefunc-fluxes.json",
    "training_data-tpipefunc-gaia.json",
    "training_data-tpipefunc-format.json",
    "training_data-tpipefunc-coverage.json",
    "training_data-tpipefunc-coordsDegrees.json",
    "training_data-tapquery.json",
    "training_data-cone.json",
    "tpipe.json",
    "tmatchn.json",
    "tmatch2.json",
    "tcopy.json",
    "tcat.json",
    "tcatn.json",
    "tapquery.json",
    "mocshape.json",
    "pixfoot.json",
    "plot2plane.json",
    "cone-examples.json",
    "training_data-tpipe6.json",
    "tapquery2.json",
    "training_data-options-desc.json",
]
EVAL_TEST_SPLIT = 0.2
BATCH_SIZE = argparser.parse_args().batch_size
LEARNING_RATE = 5e-5  # 5e-5 is a common learning rate for fine-tuning large models
NUM_EPOCHS = 5  # 1 epoch is often sufficient for pre-trained models on specific tasks.
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
if limit_mem:
    if device.type == "cuda":
        # set to 80% of total memory
        torch.cuda.set_per_process_memory_fraction(0.60)


os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", torch_dtype=DTYPE, attn_implementation="eager"
)

print("Preparing dataset...")
with open(TRAIN_FILE, "r") as f:
    data = json.load(f)

# Load additional training data if specified
if ADDITIONAL_TRAIN_FILES:
    for additional_file in ADDITIONAL_TRAIN_FILES:
        with open(additional_file, "r") as f:
            additional_data = json.load(f"{TRAINING_DATA_DIR}/{f}")
            data.extend(additional_data)
# remove any data that has "chat.jar" == "failed"
if any("chat.jar" in item for item in data):
    print(f"Total training examples before removing chat.jar fails: {len(data)}")
    data = [item for item in data if item.get("chat.jar") != "failed"]
    print(f"Total training examples after removing chat.jar fails: {len(data)}")
# set chat template for tokenizer as gemma

formatted_data = []

gemma_template = (
    "{% if messages[0]['role'] == 'system' %}"
    "{{ raise_exception('System messages are not supported by this template.') }}"
    "{% endif %}"
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{{ '<start_of_turn>' + message['role'] + '\n' + message['content'] | trim + '<end_of_turn>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<start_of_turn>model\n' }}"
    "{% endif %}"
)

tokenizer.chat_template = gemma_template

for item in data:
    chat = [
        {"role": "user", "content": item["prompt"]},
        {"role": "assistant", "content": item["response"]},
    ]
    formatted_data.append({"text": tokenizer.apply_chat_template(chat, tokenize=False)})

# formatted_data = tokenizer.apply_chat_template(data, tokenize=False)
dataset = Dataset.from_list(formatted_data)

# check a few examples
print("\nSample formatted data:")
for i in range(min(2, len(formatted_data))):  # Show first 2 examples
    print(f"\nExample {i+1}:")
    print(
        formatted_data[i]["text"][:500]
        + ("..." if len(formatted_data[i]["text"]) > 500 else "")
    )


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
    eval_strategy="steps",
    eval_steps=50,  # after how many steps to eval effects training time.
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
    # put metric data into a df and save it
    import pandas as pd

    # save training loss
    train_df = pd.DataFrame(
        {"step": train_steps, "loss": train_loss}, columns=["step", "loss"]
    )
    train_df.to_csv(f"{PLOT_OUTPUT_DIR}/train_loss.csv", index=False)

    eval_df = pd.DataFrame(
        {"step": eval_steps, "loss": eval_loss}, columns=["step", "loss"]
    )
    eval_df.to_csv(f"{PLOT_OUTPUT_DIR}/eval_loss.csv", index=False)

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

    # plt.title("Training & Evaluation Loss Curve")
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
cmd = f"python /home/rhys/llama.cpp/convert_hf_to_gguf.py {OUTPUT_DIR}/final_model --outfile {OUTPUT_DIR}/final_model.gguf"
subprocess.run(cmd, shell=True, check=True)
print("\nTraining complete and model saved in GGUF format.")

# creating 8bit version using llama.cpp quantization
print("\nQuantizing model to 8-bit GGUF format...")
cmd_8bit = f"./home/rhys/llama.cpp/build/bin/llama-quantize {OUTPUT_DIR}/final_model.gguf {OUTPUT_DIR}/final_model-Q8_0.gguf Q8_0"
subprocess.run(cmd_8bit, shell=True, check=True)
print("\n8-bit quantization complete.")

# create 4bit_K_M version using llama.cpp quantization
print("\nQuantizing model to 4-bit K_M GGUF format...")
cmd_4bit = f"./home/rhys/llama.cpp/build/bin/llama-quantize {OUTPUT_DIR}/final_model.gguf {OUTPUT_DIR}/final_model-Q4_K_M.gguf Q4_K_M"
subprocess.run(cmd_4bit, shell=True, check=True)
print("\n4-bit K_M quantization complete.")
