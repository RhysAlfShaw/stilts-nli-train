from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# Load the fine-tuned model
model_path = "./stilts-llm-finetuned/final_model"
tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto")
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
# Example inference
prompt = "## Instruction: How do I do sky matching between cat1.fits and cat2.fits?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    inputs["input_ids"],
    max_length=512,
    temperature=0.5,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
    attention_mask=inputs["attention_mask"],
    streamer=streamer,
)

# streaming the output
response = ""
for output in outputs:
    response += tokenizer.decode(output, skip_special_tokens=True)
    print(output, end="")