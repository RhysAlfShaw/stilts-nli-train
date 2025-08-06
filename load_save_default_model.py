MODEL_NAME = "meta-llama/Llama-3.2-1B"
OUTPUT_DIR = "./Llama-3.2-1B"

# load model and tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Load access token
with open("access_token", "r") as f:
    access_token = f.read().strip()
os.environ["HF_TOKEN"] = access_token


def load_default_model():

    print("Loading default model and tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

    # save the model and tokenizer

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Model and tokenizer loaded and saved successfully.")
    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = load_default_model()
    print(f"Model: {model}")
    print(f"Tokenizer: {tokenizer}")
    print("Default model loaded successfully.")
