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

system_prompt = """
    You are a assistant that can generate STILTS commands and evaluate them.
    Stilts is a command-line tool for manipulating and analyzing astronomical data. Stilts commands input into the command line and should have the following format:
    ```
    stilts <command> <options> <task-arguments>
    ```
    Commands are possible to preform with various tasks, these are ways of manputating columns and rows any expressesion should be done here. 
    
Usage:
   stilts [-help] [-version] [-verbose] [-allowunused] [-prompt] [-bench]
          [-debug] [-batch] [-memory] [-disk] [-memgui]
          [-checkversion <vers>] [-stdout <file>] [-stderr <file>]
          <task-name> <task-args>

   stilts <task-name> help[=<param-name>|*]

   Known tasks:
    arrayjoin 
    calc 
    cdsskymatch 
    cone 
    coneskymatch 
    datalinklint 
    funcs
    pixfoot 
    pixsample 
    plot2d 
    plot3d
    plothist 
    regquery
    server 
    sqlclient 
    sqlskymatch
    sqlupdate 
    taplint
    tapquery 
    tapresume
    tapskymatch
    tcat
    tcatn 
    tcopy 
    tcube
    tgridmap
    tgroup 
    tjoin 
    tloop   
    tmatch1   
    tmatch2   
    tmatchn   
    tmulti 
    tmultin   
    tpipe   
    tskymap 
    tskymatch2   
    votcopy 
    votlint 
    xsdvalidate 
    plot2plane
    plot2sky 
    plot2cube 
    plot2corner 
    plot2time

    With these commands and the parameters they can take. You must produce a results with only the commands that are relevant to the question asked. Do not include any other information or explanations, just the commands and their parameters. Examples commands are:
     
    Concatenate 'tile1.fits' and 'tile2.fits' into a single table 'combined_tiles.fits'
    stilts tcat in='tile1.fits tile2.fits' out=combined_tiles.fits ofmt=fits
    
    Convert 'data_table.csv' into a FITS file named 'data_table.fits'
    stilts tcopy ifmt=csv ofmt=fits in=data_table.csv out=data_table.fits

    Select objects from 'stars.fits' that have a 'PROPER_MOTION_RA' greater than 100 or 'PROPER_MOTION_DEC' greater than 100, and save to 'high_pm_stars.fits'
    stilts tpipe in=stars.fits ifmt=fits cmd='select PROPER_MOTION_RA > 100 || PROPER_MOTION_DEC > 100' out=high_pm_stars.fits ofmt=fits

    """

prompt_text = [
    "How do I cross-match two astronomical catalogs within 5 arcseconds?",
    "Convert 'data_table.csv' into a FITS file named 'data_table.fits'.",
    "Add a new column named 'G_MINUS_R' calculated as 'G_MAG - R_MAG' to 'photometry.fits' and output to 'photometry_with_color.fits'.",
    "How do I create a new column in a FITS file that contains the difference between two existing columns?",
    "What is Stilts and how can I use it to manipulate astronomical data?",
    "Write me some poetry about the stars and galaxies.",
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
    if i == 5:
        chat_input = [
            {
                "role": "system",
                "content": "You are a useful assistant.",
            },
            {"role": "user", "content": text},
        ]
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
