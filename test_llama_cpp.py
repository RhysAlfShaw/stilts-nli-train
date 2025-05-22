from llama_cpp import Llama
from transformers import AutoTokenizer
import time
import os

# Limit OpenMP threads to 2
os.environ["OMP_NUM_THREADS"] = "2"
# Path to your GGUF model (e.g., "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf")
MODEL_PATH = "stiltsdistil.gguf"

# Load the model.
num_threads = 5  # Adjust based on your CPU

llm = Llama(
    model_path=MODEL_PATH,
    n_threads=num_threads,  # Adjust based on your CPU
    n_batch=32,  # Reasonable batch size
    verbose=True,
    n_threads_batch=num_threads,
)
tokenizer = AutoTokenizer.from_pretrained("stilts-llm-finetuned-distilgpt2/final_model")

prompt = "How do I cross-match two astronomical catalogs within 5 arcseconds?"
template_llama = "{% for message in messages %}{{'<|start_header_id|>' + message['role'] + '<|end_header_id|>\n' + message['content'] + '<|eot_id|>' + '\n'}}{% endfor %}"
template_distilgpt = "{% for message in messages %}{{'<|start_header_id|>' + message['role'] + '<|end_header_id|>\n' + message['content'] + '<|eot_id|>' + '\n'}}{% endfor %}"

chat = [
    {"role": "system", "content": "You are a stilts command generator."},
    {"role": "user", "content": prompt},
]

# Apply chat template
formatted = tokenizer.apply_chat_template(
    chat, chat_template=template_distilgpt, tokenize=False
)

formatted += "<|start_header_id|>assistant<|end_header_id>\n"

print("Formatted chat:", formatted)
t0 = time.time()
# Generate the response
response = ""
for chunk in llm.create_completion(
    prompt=formatted, max_tokens=250, stream=True, stop=["```<|eot_id|>"]
):
    print(chunk["choices"][0]["text"], end="", flush=True)
print()  # for newline after streaming output

t1 = time.time()
print(f"Time taken: {t1 - t0:.2f} seconds")
