from llama_cpp import Llama

# Path to your GGUF model (e.g., "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf")
MODEL_PATH = "your-model.gguf"

# Load the model
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=8,  # adjust based on your CPU
    n_batch=64,  # reasonable batch size
    verbose=True,
)


# MiniGPT-style prompt wrapper (simple version)
def minigpt_prompt(user_message):
    return f"""### Instruction:
You are MiniGPT, a helpful and concise assistant.

### Input:
{user_message}

### Response:
"""


# Interactive loop
def chat():
    print("🧠 MiniGPT is ready! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            break

        prompt = minigpt_prompt(user_input)
        output = llm(prompt, max_tokens=256, stop=["###", "You:"])
        response = output["choices"][0]["text"].strip()

        print(f"MiniGPT: {response}\n")


if __name__ == "__main__":
    chat()
