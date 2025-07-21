from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

import subprocess

# read hf token from txt file
with open("access_token", "r") as f:
    hf_token = f.read().strip()

MODEL_PATH = "/scratch/Rhys/stilts_models/gemma-2b-it-finetuned/final_model"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="cuda",
    torch_dtype=dtype,
)

input_catalogue_1 = "input.fits"  # Example input catalogue
input_catalogue_2 = "input_2.fits"  # Example second input catalogue

TESTS = [
    {
        "prompt": "How can I change a fits file 'input'.fits to csv format using STILTS?",
        "input": [input_catalogue_1],
        "output": "output.csv",
    },
    {
        "prompt": "I want to do sky cross-matching between catalogues 'input.fits' and 'input_2.fits' using their RA and Dec columns and with a 6 arcsecond seperation. Call the output catalogue 'cross_matched.fits'.",
        "input": [input_catalogue_1, input_catalogue_2],
        "output": "cross_matched.fits",
    },
    {
        "prompt": "What is STILTS?",
        "input": [],
        "output": "answer.txt",
    },
    {
        "prompt": "I have duplicated rows in my catalogue 'input.fits', can you remove these and call the new catalogue 'cleaned.fits'?",
        "input": [input_catalogue_1],
        "output": "cleaned.fits",
    },
    {
        "prompt": "For my catalogue 'input.fits', I want to add a new column 'square_col' which is the square of the values in the column 'value_col'. Call the new catalogue 'squared_column.fits'.",
        "input": [input_catalogue_1],
        "output": "squared_column.fits",
    },
    {
        "prompt": "I want to filter my catalogue 'input.fits' to only include rows where the column 'value_col' is greater than 10, and save the result to 'filtered.fits'.",
        "input": [input_catalogue_1],
        "output": "filtered.fits",
    },
]
# f"I want to do sky cross-matching between catalogues {input_catalogue_1} and {input_catalogue_2} using their RA and Dec columns and with a 6 arcsecond seperation.",
# "What is STILTS?",
# f"I have duplicated rows in my catalogue ({input_catalogue_1}), can you remove these and call the new catalogue 'cleaned.fits'?",
# f"For my catalgue '{input_catalogue_1}', I want to add a new column 'square_col' which is the square of the values in the column 'value_col'.",
# f"I want to filter my catalogue '{input_catalogue_1}' to only include rows where the column 'value_col' is greater than 10, and save the result to 'filtered.fits'.",
# }

for test in TESTS:
    print("#######")
    print(f"Prompt: {test['prompt']}")
    inputs = tokenizer(
        test["prompt"],
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # take only the text between model <text> model

    response = (
        response.split("<end_of_turn>")[1].strip().split("<end_of_turn>")[0].strip()
    )
    # remove <start_of_turn>model
    response = response.replace("<start_of_turn>model", "").strip()

    print("Response:")
    print(response)
    print("#######\n")

    print("#######")

    # check if a command was generated.
    if "stilts" not in response:
        print("No STILTS command generated, skipping execution.")
        continue
    print("Running STILTS with command")

    # swap 'stilts' for '/home/rhys/stilts'
    # PATH to testing if its a valid command.
    # path_to_chat_jar = "/home/rhys/TopStiltsAgent/chat.jar"
    # import os

    # if not os.path.exists(path_to_chat_jar):
    #     print(f"Path to STILTS jar not found: {path_to_chat_jar}")

    # response_no_stilts = response.replace("stilts", "")
    # cmd = f"java -jar {path_to_chat_jar} {response.replace('stilts', '')}"
    # print(f"Command to run: {cmd}")
    # subprocess.run(
    #     cmd,
    #     check=True,
    #     text=True,
    # )

    response = response.replace("stilts", "/home/rhys/stilts")
    for input_file in test["input"]:
        response = response.replace(input_file, f"TESTING_CATALOGS/{input_file}")

    response = response.replace(test["output"], "TESTING_CATALOGS/" + test["output"])
    print(response)
    subprocess.run(response, shell=True, check=True)
    print(
        "Finished running STILTS command, check TESTING_CATALOG for the resulting output."
    )
