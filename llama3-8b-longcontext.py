from llama_cpp import Llama

num_threads = 5  # Adjust based on your CPU


llm = Llama.from_pretrained(
    repo_id="leafspark/Llama-3-8B-Instruct-Gradient-4194k-GGUF",
    filename="Llama-3-8B-Instruct-Gradient-4194k.Q4_K_M.fixed.gguf",
    n_threads=num_threads,  # Adjust based on your CPU
)

x = llm.create_chat_completion(
    messages=[{"role": "user", "content": "What is the capital of France?"}]
)

print(x)
