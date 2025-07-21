from llama_cpp import Llama

llm = Llama(
    model_path="./llama3.2-1b-stilts.gguf", n_threads=4, n_ctx=2048, n_batch=512
)
response = llm.create_chat_completion(
    messages=[
        {
            "role": "system",
            "content": "You are a stilts command generator.",
        },
        {
            "role": "user",
            "content": "Can you make a stilts command that does catalogue matching?",
        },
    ],
    stop=["<|eot_id|>"],
    temperature=0.7,
    top_p=0.9,
    max_tokens=1000,
    stream=True,  # Enable streaming
)

for chunk in response:
    content = chunk["choices"][0]["delta"].get("content", "")
    print(content, end="", flush=True)
print()
