from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

# read hf token from txt file
with open("access_token", "r") as f:
    hf_token = f.read().strip()

model_id = "google/gemma-7b-it"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype=dtype,
)

context_which_command = """
Yes, I will give you the best STILTS command you should use to preform the task that you describe. I will only give you the command, no other text. The command will be in a code block.
The commands and their descriptions are as follows:

    - tpipe: This command is used to process data streams with various operations like filtering, transforming, and outputting results in different formats.
    - tcopy: This command is used to copy data from one format to another, allowing for format conversion of tables.
    - tcat: This command concatenates multiple tables into a single table, with options for input formats, output formats, and various operations on the data.
    - tmatch2: This command performs cross-matching between two tables, allowing for complex matching criteria,
      output options, and various parameters to control the matching process.
"""
commands = {
    "tpipe": "tpipe ifmt=<in-format> istream=true|false cmd=<cmds> omode=out|meta|stats|count|checksum|cgi|discard|topcat|samp|plastic|tosql|gui out=<out-table> ofmt=<out-format> [in=]<table>",
    "tcopy": "tcopy ifmt=<in-format> ofmt=<out-format> [in=]<table> [out=]<out-table>",
    "tcat": "tcat in=<table> [<table> ...] ifmt=<in-format> multi=true|false istream=true|false icmd=<cmds> ocmd=<cmds> omode=out|meta|stats|count|checksum|cgi|discard|topcat|samp|plastic|tosql|gui out=<out-table> ofmt=<out-format> seqcol=<colname> loccol=<colname> uloccol=<colname> lazy=true|false countrows=true|false",
    "tmatch2": """tmatch2 ifmt1=<in-format> ifmt2=<in-format> icmd1=<cmds> icmd2=<cmds>
               ocmd=<cmds>
               omode=out|meta|stats|count|checksum|cgi|discard|topcat|samp|plastic|tosql|gui
               out=<out-table> ofmt=<out-format> matcher=<matcher-name>
               values1=<expr-list> values2=<expr-list> params=<match-params>
               tuning=<tuning-params>
               join=1and2|1or2|all1|all2|1not2|2not1|1xor2
               find=all|best|best1|best2 fixcols=none|dups|all suffix1=<label>
               suffix2=<label> scorecol=<col-name>
               progress=none|log|time|profile
               runner=parallel|parallel<n>|parallel-all|sequential|classic|partest
               [in1=]<table1> [in2=]<table2>""",
}
examples = {
    "tpipe": """```bash
stilts tpipe ifmt=csv istream=true cmd="filter 'col1 > 10' transform 'col2 = col1 * 2'" omode=out out=result.csv ofmt=csv in=input.csv
```""",
    "tcopy": """```bash
stilts tcopy ifmt=csv ofmt=tsv in=input.csv out=result.tsv
```""",
    "tcat": """```bash
stilts tcat in=input1.csv in=input2.csv ifmt=csv multi=true istream=true
           ocmd="addcol 'newcol = col1 + col2'" omode=out out=result.csv ofmt=csv
```""",
    "tmatch2": """```bash
stilts tmatch2 ifmt1=csv ifmt2=csv icmd1="
filter 'col1 > 10'" icmd2="filter 'col2 < 20'"
              ocmd="addcol 'score = col1 - col2'" omode=out
              out=result.csv ofmt=csv matcher=skyvalues
              values1="ra,dec" values2="ra,dec" params="maxsep=5"
              join=1and2 find=best scorecol=score progress=log
              in1=catalog1.csv in2=catalog2.csv
```""",
}


# command = "tmatch2"
def command_context(command):
    context = f"""
    Yes, I am a STILTS command generator. I will be given a prompt to generate a STILTS command:

    The command I can generate from is {command}

    The possible parameters for the command are:
    {commands[command]}

    An example of the command is:
    {examples[command]}

    """
    return context


prompt = "Can you make a STILTS command to cross-match two catalogs within 5 arcseconds? Please provide the command in a code block."


chat = [
    {"role": "user", "content": "Hello, can you help me with STILTS commands?"},
    {"role": "assistant", "content": context_which_command},
    {
        "role": "user",
        "content": "Great, for this prompt" + prompt + " which command should I use?",
    },
]

prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids)
print(""" Print the generated function""")
print(tokenizer.decode(outputs[0]))


# now create command from now we know which command to use.

chat = [
    {"role": "user", "content": "Hello, can you help me with STILTS commands?"},
    {
        "role": "assistant",
        "content": command_context("tmatch2"),
    },  # for testing we assume tmatch2.
    {"role": "user", "content": prompt},
]

prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    **input_ids, max_new_tokens=200, do_sample=False, temperature=0.1
)
print(""" Print the generated command""")
print(tokenizer.decode(outputs[0]))
