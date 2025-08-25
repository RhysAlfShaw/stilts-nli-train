import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import json
from model import Model

# --- 1. Model and Tokenizer Setup (Your code was good here) ---
model_path = "meta-llama/Llama-3.2-3B-Instruct"  # Using 8B as a stand-in, logic is identical for 1B
# meta-llama/Llama-3.2-3B-Instruct
print(f"Loading model: {model_path}")
import os

STILTS_model = Model()

with open("access_token", "r") as f:
    access_token = f.read().strip()
os.environ["HF_TOKEN"] = access_token

# Note: Quantization might affect tool-calling performance. For best results, use float16 if VRAM allows.
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,  # bfloat16 is often better than float16
    device_map="auto",
    # For smaller models like 1B, you might not need quantization
    # load_in_4bit=True
)

tokenizer = AutoTokenizer.from_pretrained(model_path)
# Llama3.2 uses a specific EOT token as the pad token
tokenizer.pad_token = tokenizer.eos_token


def get_stilts_command(description: str):
    command = STILTS_model.generate_stream(description)
    full_chunks = ""
    for chunk in command:
        print(chunk, end="", flush=True)
        full_chunks += chunk
    return full_chunks


available_functions = {
    "stilts_command_generation": get_stilts_command,
}


system_prompt = """
You are a chatbot designed to assist with generating STILTS commands based on user descriptions. They should provide a task description input and output files names.
STILTS is a command-line tool for manipulating and analyzing astronomical data. 

If you are asked what stilts is, you can reply with something like:
"STILTS (Starlink Tables Infrastructure Library for Tables) is a command-line tool designed for manipulating and analyzing astronomical data. It provides a wide range of functionalities for working with tabular data, including filtering, sorting, joining, and plotting. STILTS is particularly useful for astronomers and astrophysicists who need to process large datasets efficiently."

If you are asked what you can do or what tasks you support reply with the following tasks:
tpipe, tcat, tmatch2, tcopy.


You must decide if you should reply with text normally or reply with only a function call.

You are an expert in composing functions. If you are given a question or decription of a stilts command and a set of possible functions. 
Based on the question, you will need to make one or more function/tool calls to achieve the purpose. 
If none of the function can be used, point it out. If the given question lacks the parameters required by the function,
also point it out. You should only return the function call in tools call sections.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]\n
You SHOULD NOT include any other text in the response.
You MUST NOT under any circumstances return any functions like this: [func_name1(parameters={'params_name_1': 'params_value_1', 'params_name_2': 'params_value_2'})]\n
It must always be in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]\n
Here is a list of functions in JSON format that you can invoke:

[
    {
        "name": "stilts_command_generation",
        "description": "Generates a Stilts command for an LLM agent to execute based on the provided description.",
        "parameters": {
            "type": "dict",
            "requied": ["properties"],
            "properties": {
                "type": "string",
                "description": "A text description of the task for which a Stilts command is needed in natural language not in code.",
            },
        },
    }
]

Should you decide to return the function call(s), Put it in the format of [func1(params_name=params_value, params_name2=params_value2...), func2(params)]
NO other text MUST be included.

If you do not need to call any function, reply normally. 

"""


user_prompt_text = input(">> ")
# "Can you create a Stilts command that will select rows from input.fits table where the column 'flux' is greater than 10 and save the result to 'output.fits'?"
# user_prompt_text = "Hello! How are you?"
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt_text},
]

prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.1,  # Lower temperature for more deterministic output
    top_p=0.95,
)

response_text = tokenizer.decode(
    outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
)

print(f"Response: {response_text}")


if "stilts_command_generation" in response_text:
    # Extract the function call from the response
    try:
        description = response_text.split("properties='")[1].split("')")[0].strip("'\"")
    except IndexError:
        try:
            description = (
                response_text.split("description=")[1].split(")")[0].strip("'\"")
            )
        except IndexError:
            try:
                description = (
                    response_text.split("type='")[1].split("')")[0].strip("'\"")
                )
            except IndexError:
                print("Error: processing llm tool call response")

    print(f"Extracted description: {description}")
    command = available_functions["stilts_command_generation"](description)
    print(f"Generated Stilts command: {command}")


print("--- Generated command --- \n\n {command}")
user = (
    input(
        "If you want to execute this command, type 'r', or to modify it, type 'e' or try and generate it again, type 'g': "
    )
    .strip()
    .lower()
)

if user == "r":
    print(f"Executing command: {command}")
    # Here you would execute the command, e.g., using subprocess or similar
    # subprocess.run(command, shell=True)
elif user == "e":
    command = input("Enter a new description for the Stilts command: ")

elif user == "g":
    print("Generating a new command...")
    # You can loop back to the command generation logic here
