import shlex
import re
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import shlex
import re
from sentence_transformers import SentenceTransformer, util
import os
import numpy as np


print("Loading sentence-transformer model...")
model_embed = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded successfully.")

device = "cuda"

TEST_CASE_DIR = "DATA/TEST_CASES/"
# get path for all .json in test_case_dir
test_data_files = [
    f"{TEST_CASE_DIR}{file}"
    for file in os.listdir(TEST_CASE_DIR)
    if file.endswith(".json")
]

TEST_CASES = []
for file in test_data_files:
    with open(file, "r") as f:
        data = json.load(f)
        TEST_CASES.extend(data)


model_path = "/scratch/Rhys/stilts_models/gemma-2b-finetuned/final_model"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="cuda",
    attn_implementation="eager",
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


def get_llm_response(prompt: str) -> str:
    streamer = generate_stream_transformers(prompt, max_new_tokens=500)
    generated_text = ""
    # print("\n--- Generated Response ---")
    for new_text in streamer:
        # print(new_text, end="", flush=True)
        generated_text += new_text
    return generated_text


def evaluate_command_similarity(generated: str, expected: str) -> float:
    """
    Calculates the semantic similarity of two commands using vector embeddings.
    Returns a score between 0.0 and 1.0.
    """
    if not generated or not expected:
        return 0.0

    # Encode both command strings into vectors
    embedding1 = model_embed.encode(generated, convert_to_tensor=True)
    embedding2 = model_embed.encode(expected, convert_to_tensor=True)

    # Compute cosine similarity
    cosine_score = util.cos_sim(embedding1, embedding2)

    # The result is a tensor, so we extract the float value
    return cosine_score.item()


def main():
    """Runs all tests and generates a final report."""
    print(
        "\nStarting STILTS Command Generation Test Suite (Embedding Similarity Mode)..."
    )
    results = []
    from tqdm import tqdm

    for i, case in tqdm(
        enumerate(TEST_CASES), total=len(TEST_CASES), desc="Running tests"
    ):

        generated_command = get_llm_response(case["prompt"])
        score = evaluate_command_similarity(generated_command, case["expected"])
        results.append({**case, "generated": generated_command, "score": score})

    print("\n" + "=" * 80)
    print("TEST REPORT")
    print("=" * 80)

    low_scoring = [res for res in results if res["score"] < 0.95]
    if low_scoring:
        print("\n CASES WITH SIMILARITY SCORE < 95%:\n")
        for res in sorted(low_scoring, key=lambda x: x["score"]):
            print(f"  Score:     {res['score']:.2f}")
            print(f"  Category:  {res['category']}")
            print(f"  Prompt:    {res['prompt']}")
            print(f"  Expected:  `{res['expected']}`")
            print(f"  Generated: `{res['generated']}`\n")
        print("-" * 80)
    else:
        print("\n All generated commands had a similarity score of 95% or higher!\n")
        print("-" * 80)
    # save results to results.json
    import json

    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)

    category_scores = {}
    for res in results:
        cat = res["category"]
        if cat not in category_scores:
            category_scores[cat] = {"scores": [], "total": 0}
        category_scores[cat]["scores"].append(res["score"])
        category_scores[cat]["total"] += 1

    print("\n PERFORMANCE SUMMARY:\n")
    total_scores = []
    total_std = []
    for category, data in sorted(category_scores.items()):
        avg_score = (sum(data["scores"]) / data["total"]) * 100
        std_of_score = np.std(data["scores"])
        total_std.append(std_of_score)
        total_scores.extend(data["scores"])
        print(
            f"  - {category:<25} | Cases: {data['total']:<2} | Avg. Similarity: ({avg_score:6.2f} ± {std_of_score:.2f}) %"
        )

    print("-" * 80)
    overall_avg = (sum(total_scores) / len(total_scores)) * 100
    overall_std = np.std(total_scores)
    print(
        f"  OVERALL AVERAGE SIMILARITY:{'':<8} | Cases: {len(total_scores):<2} | Avg. Similarity: ({overall_avg:6.2f}±{overall_std:6.2f})%\n"
    )


if __name__ == "__main__":
    main()
