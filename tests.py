print("\nRunning example inference...")

model_path = "/scratch/Rhys/stilts_models/gemma-2b-finetuned/final_model"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import subprocess

GREEN = "\033[92m"
RESET = "\033[0m"
RED = "\033[91m"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # Use float16 for better performance on GPUs
    low_cpu_mem_usage=True,  # Reduce memory usage during loading
    device_map="auto",  # Automatically map model to available
).to("cuda" if torch.cuda.is_available() else "cpu")

device = model.device

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    use_fast=True,  # Use fast tokenizer for better performance
    padding_side="right",  # Ensure padding is on the right side
)

model.eval()  # Set model to evaluation mode

prompt_text = [
    "What is STILTS? please explain in detail.",
    "How can I match catalogue `EUCLID.fits` with `GAIA.fits` using a 1 arcsecond radius?",
    "Match a local table with a remote table from VizieR at 'http://vizier.u-strasbg.fr/viz-bin/VizieR-3?-source=I/345/gaia2' using a 1 arcsecond radius and save the result to 'matched.fits'.",
    "I want to select sources within my catalogue 'cat.fits' that are within a skydistance of 10 arcseconds.",
    "Filter my catalog 'my_objects.fits' to find all sources that fall within the SDSS DR9 footprint, using its VizieR MOC identifier. Output to 'sdss_crossmatch.fits'.",
    "Select all the sourcce in 'observations.fits' where all columns are not null and save it to 'cleaned.fits'.",
    "create a STILTS command that will change any missing or nan values to -99",
    "Make a new column in catalogue input.fits that is defined as column MAG_u - MAG_i and save as a parquet.",
    "Combine two post-processing commands: first, select sources with G magnitude brighter than 12, then keep only the ID and position columns.",
    "Give me statistics on my catalogue 'DSSD_1232lf.fits'.",
    "Give me an example STILTS command I could run on my catalogue 'Deepmind.fits'.",
    "Select from my catalogue 'sources.fits' all the rows that do not have null values and save them as 'non_null_sources.fits'.",
    "What stilts tasks are you trained on?",
    "I want to remove all rows that do not have any column with a value less than 10. Please save this as 'cleaned.fits`.",
]

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="cuda",
    attn_implementation="eager",
    # quantization_config=quantization_config,
)

tokenizer = AutoTokenizer.from_pretrained(model_path)


def generate_stream_transformers(prompt: str, max_new_tokens: int = 500):
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    prompt_templated = (
        "<bos><start_of_turn>user\n"
        + prompt
        + "<end_of_turn>\n<start_of_turn>assistant\n"
    )

    inputs = tokenizer(prompt_templated, return_tensors="pt").to(device)
    eos_token_id = tokenizer("<end_of_turn>")["input_ids"][-1]

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.3,
        top_p=0.95,
        pad_token_id=eos_token_id,
        eos_token_id=eos_token_id,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    return streamer


def check_cmd(cmd):
    # remove stilts from the response
    cmd = cmd.replace("stilts", "").strip()
    base_command = "docker run stilts-chat-stilts-dev-env java -jar chat.jar"
    cmd = f"{base_command} {cmd}"
    try:
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True
        )
        print(f"\n Command executed {GREEN}successfully{RESET}:\n\n {cmd}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n Command {RED}failed{RESET}:\n\n {cmd}\nError: {e.stderr}")
        return False


for i, text in enumerate(prompt_text):
    print(f"Prompt {i + 1}: {text}")
    streamer = generate_stream_transformers(text, max_new_tokens=500)
    generated_text = ""
    print("\n--- Generated Response ---")
    for new_text in streamer:
        print(new_text, end="", flush=True)
        generated_text += new_text

    check_cmd(generated_text)
    print("\n")
    print("------------------------")
    print("\n")
