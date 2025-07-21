print("\nRunning example inference...")

# model_path = "stilts-llm-finetuned-distilgpt2/final_model" # DistilGPT-2 model path
model_path = "meta-llama/Llama-3.2-1B-Instruct"  # Llama 3 model path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # Use float16 for better performance on GPUs
    low_cpu_mem_usage=True,  # Reduce memory usage during loading
    device_map="auto",  # Automatically map model to available devices
).to("cuda" if torch.cuda.is_available() else "cpu")

device = model.device

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    use_fast=True,  # Use fast tokenizer for better performance
    # padding_side="right",  # Ensure padding is on the right side
)

model.eval()  # Set model to evaluation mode

functions = [
    {
        "name": "tpipe",
        "description": "Does opertations on a catalogue.",
        "parameters": {
            "type": "object",
            "properties": {
                "ifmt": {
                    "type": "string",
                    "description": "Input format of the table.",
                },
                "istream": {
                    "type": "boolean",
                    "description": "Whether the input is a stream.",
                },
                "cmd": {
                    "type": "string",
                    "description": "Commands to execute on the table.",
                },
                "omode": {
                    "type": "string",
                    "description": "Output mode for the table.",
                    "enum": [
                        "out",
                        "meta",
                        "stats",
                        "count",
                        "checksum",
                        "cgi",
                        "discard",
                        "topcat",
                        "samp",
                        "plastic",
                        "tosql",
                        "gui",
                    ],
                },
                "out": {
                    "type": "string",
                    "description": "Output table name.",
                },
                "ofmt": {
                    "type": "string",
                    "description": "Output format of the table.",
                },
                "in": {
                    "type": "string",
                    "description": "Input table name.",
                    "default": "",
                },
            },
        },
    },
]

system_prompt = f""" 
    You are a helpful assistant that can generate STILTS commands for manipulating astronomical data. "
    You can only generate stilts commands that are defined in the functions list below.
    Task list: {functions}
    """
prompt_text = [
    "I want to make a new column in a table that is the sum of two other columns.",
]

for i, text in enumerate(prompt_text):
    print(f"Prompt {i + 1}: {text}")

    chat_input = [
        {
            "role": "system",
            "content": f"{system_prompt}",
        },
        {"role": "user", "content": text},
    ]
    print("\n--- Chat Input ---")
    print(chat_input)
    print("------------------------")
    template_distilgpt2 = "{% for message in messages %}{{'<|start_header_id|>' + message['role'] + '<|end_header_id|>\n' + message['content'] + '<|eot_id|>' + '\n'}}{% endfor %}"

    # Apply chat template for inference
    # We add add_generation_prompt=True here because we want the model to generate the assistant's response
    input_text = tokenizer.apply_chat_template(
        chat_input,
        chat_template=template_distilgpt2,  # Use custom chat template for Llama 3
        tokenize=False,
        add_generation_prompt=True,  # Important for inference
    )
    # append the assistant role to the input text
    input_text += "<|start_header_id|>assistant<|end_header_id|>\n"
    # print(f"\n--- Input Text for Model ---\n{input_text}\n------------------------")
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    eos_token_ids = [tokenizer.eos_token_id]

    if tokenizer.vocab.get("<|eot_id|>") is not None:  # Llama3 specific end of turn
        eos_token_ids.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))
    if (
        tokenizer.vocab.get("<|end_of_text|>") is not None
    ):  # Llama3 specific end of sequence
        eos_token_ids.append(tokenizer.convert_tokens_to_ids("<|end_of_text|>"))

    # Remove duplicates
    eos_token_ids = sorted(list(set(eos_token_ids)))

    # print(f"Using EOS token IDs: {eos_token_ids}")

    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_token_ids,  # Can be a list of EOS token IDs for Llama 3
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.2,
    )

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            generation_config=generation_config,
        )

    # Decode the generated output

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)

    # remove everything that is not in assistant<|end_header_id|>... <|eot_id|>

    generated_text = generated_text.split("assistant<|end_header_id|>")[-1].strip()
    generated_text = generated_text.split("<|eot_id|>")[0].strip()
    print("\n--- Generated Response ---")
    print(generated_text)
    print("\n")
    print("------------------------")
    print("\n")
