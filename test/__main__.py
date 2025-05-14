# from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
# import torch

# model_path = "./stilts-llm-finetuned-distilgpt2/final_model"

# tokenizer = AutoTokenizer.from_pretrained(model_path)

# print("Loading standard fine-tuned model...")
# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     device_map="auto",
#     torch_dtype=torch.float16,
# )

# # Create streamer
# streamer = TextStreamer(
#     tokenizer,
#     skip_prompt=True,
# )

# # Example usage
# prompt = "How do I cross-match two astronomical catalogs within 5 arcseconds?"
# template = "{% for message in messages %}{{'<|start_header_id|>' + message['role'] + '<|end_header_id|>\n' + message['content'] + '<|eot_id|>' + '\n'}}{% endfor %}"

# chat = [
#     {
#         "role": "system",
#         "content": "You are a stilts command generator.",
#     },
#     {"role": "user", "content": prompt},
# ]

# formatted = tokenizer.apply_chat_template(chat, chat_template=template, tokenize=False)

# # add the assistent role header

# formatted += "<|start_header_id|>assistant<|end_header_id>\n"
# print(formatted)
# print("\nGenerating response...")
# inputs = tokenizer(
#     formatted,
#     return_tensors="pt",
# ).to(model.device)

# eot_token_id = tokenizer.eos_token_id

# outputs = model.generate(
#     **inputs,
#     streamer=streamer,
#     max_new_tokens=512,
#     eos_token_id=eot_token_id,
#     do_sample=True,
#     temperature=0.7
# )

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
import time

model_path = "./stilts-llm-finetuned-distilgpt2/final_model"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("Loading standard fine-tuned model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
)

# # Ensure <|endoftext|> is recognized as a special token
# eot_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"

# Create streamer
streamer = TextStreamer(
    tokenizer,
    skip_prompt=False,
    skip_special_tokens=True,
)

# Chat formatting
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
# Tokenize input
inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
# assert (
#     "<|endoftext|>" in tokenizer.get_vocab()
# ), "Your model/tokenizer was not trained with <|endoftext|>"

# Generate response
print("\nGenerating response...\n")
time_end = time.time()
output = model.generate(
    **inputs,
    # streamer=streamer,
    max_new_tokens=500,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    do_sample=True,
    temperature=0.7
)

# Decode and print the generated response

decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
# print("Decoded output:", decoded_output)
print("GPU Inference time:", time.time() - time_end, " seconds")
cropped_output = decoded_output.split("<|start_header_id|>assistant<|end_header_id>")[
    1
].split("<|eot_id|>")[0]

# print("Cropped output:", cropped_output)

# create an interator class that will take a string and yield it in chunks seperated by spaces
# def chunk_string(s, chunk_size):


class ChunkString:
    def __init__(self, s):
        self.s = s
        self.index = 0
        self.chunks = self.s.split(" ")

    def __iter__(self):
        return self

    def __next__(self):
        # add a waiting time of 0.1 seconds
        import time

        time.sleep(0.05)
        if self.index >= len(self.chunks):
            raise StopIteration
        chunk = self.chunks[self.index]

        self.index += 1
        return chunk


# example usage
for chunk in ChunkString(cropped_output):
    print(chunk, end=" ", flush=True)

# print("\nGenerated response:")
# print(tokenizer.decode(output[0], skip_special_tokens=True))

# now do CPU inference
# limit number of cpu threads to 4

model_cpu = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float32,  # use float32 for CPU inference
).to("cpu")

# Create streamer
streamer = TextStreamer(
    tokenizer,
    skip_prompt=False,
    skip_special_tokens=True,
)

inputs = tokenizer(formatted, return_tensors="pt").to(model_cpu.device)

# Generate response
print("\nGenerating response...\n")
time_end = time.time()
torch.set_num_threads(1)

output = model_cpu.generate(
    **inputs,
    # streamer=streamer,
    max_new_tokens=500,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    do_sample=True,
    temperature=0.7
)

# time_end = time.time()

print("CPU Inference time:", time.time() - time_end, " seconds")
# Decode and print the generated response
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
# decoded_output = output[0]["generated_text"]
# print("Decoded output:", decoded_output)
print("CPU Inference time:", time.time() - time_end, " seconds")
cropped_output = decoded_output.split("<|start_header_id|>assistant<|end_header_id>")[
    1
].split("<|eot_id|>")[0]
# print("Cropped output:", cropped_output)
print("\nGenerated response:")
print(cropped_output)
