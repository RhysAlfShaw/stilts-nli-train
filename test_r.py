from llama_cpp import Llama

# Download and load a GGUF model directly from Hugging Face
llm = Llama.from_pretrained()
response = llm.create_chat_completion(
    messages=[{"role": "user", "content": "How does a black hole work?"}]
)
print(response)
