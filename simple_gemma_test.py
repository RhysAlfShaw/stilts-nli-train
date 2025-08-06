from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

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

Prompts = [
    "How can I change a fits file (testing.fits) to csv format using STILTS?",
    "I want to do sky cross-matching between catalogues DRUID.fits and pybdsf.fits using their RA and Dec columns and with a 6 arcsecond seperation.",
    "What is STILTS?",
    "I have duplicated rows in my catalogue based on column 'RA', can you remove these and call the new catalogue 'cleaned.fits'?",
    "For my catalgue 'input.fits', I want to add a new column 'square_col' which is the square of the values in the column 'value_col'.",
    "I want to filter my catalogue 'input.fits' to only include rows where the column 'value_col' is greater than 10, and save the result to 'filtered.fits'.",
]

for Prompt in Prompts:
    print("#######")
    print(f"Prompt: {Prompt}")
    inputs = tokenizer(
        Prompt,
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # take only the text between model <text> model
    response = response.split("model")[1].split("model")[0].strip()
    print("Response:")
    print(response)
    print("#######\n")
