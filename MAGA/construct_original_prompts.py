'''
When constructing original prompts,
auxiliary information about which model the prompt is assigned to for generation is also annotated based on MAGA's strategy.
'''

import argparse
import itertools
from uuid import uuid4
from datasets import load_dataset

ORIGINAL_PROMT_TEMPLATE = {
    "Reddit": "Write just the body of a Reddit post titled \"{title}\". Do not repeat the title.",
    "S2ORC": "Write the abstract for the scientific paper titled \"{title}\". It is preferable not to start with \"This paper\".",
    "Wikipedia": "Write the body of a Wikipedia article titled \"{title}\".",
    "wikiHow": "Write the body of a wikiHow article titled \"{title}\".",
    "Trustpilot Reviews": "Write the body of a Trustpilot review titled \"{title}\". Do not give it a title.",
    "Amazon Reviews": "Write the body of an Amazon review titled \"{title}\". Do not give it a title.",
    "Yahoo Answers": "Write just the response to the question titled \"{title}\" on Yahoo Answers. Do not repeat the question.",
    "Natural Questions": "Provide the answer to the question \"{title}\".",
    "CC News": "Write the body of a news article titled \"{title}\". Do not repeat the title.",
    "NPR News": "Write the body of a NPR news article titled \"{title}\". Do not repeat the title.",
}

MODEL_LIST = [
    "GPT-4o-mini",
    "Gemini-2.0-flash",
    "DeepSeek-V3",
    "Qwen3-plus",
    "Mistral-Medium",
    "Hunyuan-TurboS",
    "Llama-3.1-8B-Instruct",
    "gemma-3-12b-it",
    "DeepSeek-R1-0528-Qwen3-8B",
    "Qwen3-8B",
    "Ministral-8B-Instruct-2410",
    "Hunyuan-7B-Instruct"
]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, required=True, help="the sampled MAGA human data file path")
    parser.add_argument("-o", "--output_path", type=str, required=True)
    parser.add_argument("--sort-domain", action="store_true",
                        help="according to MAGA's strategy, data from each domain needs to be kept together so that it can be evenly divided among the models for generation. Under the normal MAGA process, data from each domain is already together when merged, so there is no need to re-sort, so this argument can be turned off")
    return parser.parse_args()


def process(examples: dict[str, list]) -> dict[str, list]:
    prompts = []
    for title, domain in zip(examples["title"], examples["domain"]):
        prompt_template = ORIGINAL_PROMT_TEMPLATE[domain]
        prompt = prompt_template.format(title=title)
        prompts.append(prompt)
    examples["prompt_id"] = [str(uuid4()) for _ in range(len(prompts))]
    examples["system_prompt"] = [None for _ in range(len(prompts))]
    examples["user_prompt"] = prompts

    # According to MAGA's strategy, each domain is evenly divided among various models for generation.
    model_names = []
    data_amount_each_domain = [len(list(group)) for _, group in itertools.groupby(examples["domain"])]
    for num in data_amount_each_domain:
        for model_name in MODEL_LIST:
            model_names += [model_name] * (num // len(MODEL_LIST))
    examples["model"] = model_names

    return examples


if __name__ == '__main__':
    args = parse_arguments()

    print(f"Loading sampled MAGA human dataset in directory {args.input_path}...")
    dataset = load_dataset("json", data_files=args.input_path)["train"]

    if args.sort_domain:
        dataset = dataset.sort("domain")

    dataset = dataset.map(process, batched=True, batch_size=None, remove_columns=["text"])

    print(f"Writing original prompt dataset to output path: {args.output_path}...")
    dataset.to_json(args.output_path, orient="records", lines=True)

    print(f"Done!")
