from llama_cpp import Llama
from transformers import AutoTokenizer
import time
import os

# Limit OpenMP threads to 2
# Path to your GGUF model (e.g., "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf")
# MODEL_PATH = "stiltsdistil.gguf"
MODEL_PATH = "MODELS/llama3.2-1b-instruct.gguf"
# Load the model.
num_threads = 5  # Adjust based on your CPU

llm = Llama(
    model_path=MODEL_PATH,
    n_threads=num_threads,  # Adjust based on your CPU
    n_batch=32,  # Reasonable batch size
    verbose=True,
    n_threads_batch=num_threads,
)
# tokenizer = AutoTokenizer.from_pretrained("llama3.2-1b-instruct")

# prompt = "Hello, Can you make me a peom."
# template_llama = "{% for message in messages %}{{'<|start_header_id|>' + message['role'] + '<|end_header_id|>\n' + message['content'] + '<|eot_id|>' + '\n'}}{% endfor %}"
# # template_distilgpt = "{% for message in messages %}{{'<|start_header_id|>' + message['role'] + '<|end_header_id|>\n' + message['content'] + '<|eot_id|>' + '\n'}}{% endfor %}"

# chat = [
#     {"role": "user", "content": prompt},
# ]

# # Apply chat template
# formatted = tokenizer.apply_chat_template(
#     chat, chat_template=template_llama, tokenize=False
# )

# formatted += "<|start_header_id|>assistant<|end_header_id>\n"

# print("Formatted chat:", formatted)
# t0 = time.time()
# # Generate the response
# response = ""
# for chunk in llm.create_completion(
#     prompt=formatted, max_tokens=250, stream=True, stop=["<|eot_id|>"]
# ):
#     print(chunk["choices"][0]["text"], end="", flush=True)
# print()  # for newline after streaming output

# t1 = time.time()
# print(f"Time taken: {t1 - t0:.2f} seconds")
prompt = "Hello, Can you make me a poem?"
messages = [
    # {
    #     "role": "system",
    #     "content": "You are a helpful assistant. That will take context and use it to answer the question about the documentation it is about. Also say which of the context items you used to answer the question. If the user wants you to create a command, this must be passed off to the fine-tuned model.",
    # },
    {"role": "user", "content": prompt},
]
print(f"Generating response for prompt: {prompt}")

# Call the chat completion method with the stream parameter
response_generator = llm.create_chat_completion(
    messages=messages,
    max_tokens=250,
    # temperature=mperature,
    # top_p=top_p,/
    stop=["<|eot_id|>"],
    stream=True,  # Control whether to stream the response
)
for chunk in response_generator:
    print(chunk["choices"][0]["delta"].get("content", ""), end="", flush=True)
print()  # for newline after streaming output
