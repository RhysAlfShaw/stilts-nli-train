import argparse
from transformers import AutoTokenizer
from llama_cpp import Llama
import sys
import readline
import time
import os
import sys
import readline
import subprocess


MODEL_PATH = "stiltsdists.gguf"
# MODEL_PATH = "llama3.2-1b.gguf"  # Adjust this to your model path
# MODEL_PATH = "stiltsdistil.gguf"
# MODEL_PATH = "llama3.2-1b-stilts.gguf"  # Adjust this to your model path
num_processes = 5  # Adjust based on your CPU.

template_distilgpt = "{% for message in messages %}{{'<|start_header_id|>' + message['role'] + '<|end_header_id|>\n' + message['content'] + '<|eot_id|>' + '\n'}}{% endfor %}"


def input_with_prefill(prompt, text):
    def hook():
        readline.insert_text(text)
        readline.redisplay()

    readline.set_pre_input_hook(hook)
    try:
        return input(prompt)
    finally:
        readline.set_pre_input_hook()


def load_llm():
    llm = Llama(
        model_path=MODEL_PATH,
        n_threads=num_processes,  # Adjust based on your CPU
        n_batch=64,  # Reasonable batch size
        verbose=False,
        n_threads_batch=num_processes,
        n_ctx=131072,  # Adjust based on your model's context length
    )
    # tokenizer = AutoTokenizer.from_pretrained(

    # "stilts-llm-finetuned-distilgpt2/final_model"
    # )

    return llm


# @profile
def main():
    parser = argparse.ArgumentParser(description="A template for a CLI Python program.")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "prompt", nargs="?", help="The prompt to generate STILTS commands."
    )
    parser.add_argument(
        "-n_threads", type=int, default=num_processes, help="Threads for inference"
    )

    parser.add_argument(
        "-d",
        help="Attach a directory location to the command, so the model can see files.",
        default=os.getcwd(),
    )

    args = parser.parse_args()

    if not args.prompt:
        print(
            "No prompt provided. Please provide a prompt to generate stilts commands."
        )
        exit(1)

    prompt = args.prompt.strip()

    chat = [
        {"role": "system", "content": "You are a stilts command generator."},
        {"role": "user", "content": prompt},
    ]

    # if -d in args:
    if args.d:
        chat = [
            {
                "role": "system",
                "content": "You are a stilts command generator.",
            },
            {"role": "user", "content": prompt},
        ]

    llm = load_llm()

    response = ""

    print("\nGenerating command...\n")
    print(chat)
    response_stream = llm.create_chat_completion(
        messages=chat,
        temperature=0.1,
        top_p=0.1,
        max_tokens=200,
        stream=True,  # Enable streaming
    )

    for chunk in response_stream:
        content = chunk["choices"][0]["delta"].get("content", "")
        response += content
        print(content, end="", flush=True)

    # not terminating the stream

    ll_fix = Llama(
        model_path="lllama3.2-1b-instruct.gguf",
        n_threads=args.n_threads,  # Adjust based on your CPU
        n_batch=64,  # Reasonable batch size
        verbose=False,
        n_threads_batch=args.n_threads,
    )
    response = ll_fix.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You fix output from other LLMs You will be given a cli tool command You need to clean it up for execution. The command needs to be of the form `stilts <command> <args>` and should not contain any other text or comments. Rmove repeated arguements or those that are not needed.",
            },
            {
                "role": "user",
                "content": f"Original prompt: {chat}, Responce: {response}",
            },
        ],
        temperature=0.1,
        top_p=0.1,
        max_tokens=200,
    )

    print("\n\nCommand generated:\n")
    print(response["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
