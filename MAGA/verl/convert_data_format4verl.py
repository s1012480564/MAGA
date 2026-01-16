import os
import argparse
from datasets import load_dataset


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--maga_prompt_dataset_path", type=str, required=True,
                        help="Pass in dataset in the form processed by construct_original_prompts.py. It's OK to have it processed further.")
    parser.add_argument("-o", "--output_path", type=str, required=True, help="parquet type needed.")
    return parser.parse_args()


def process(examples: dict[str, list]) -> dict[str, list]:
    length = len(examples["title"])
    system_prompts = examples["system_prompt"]
    user_prompts = examples["user_prompt"]

    examples["data_source"] = [None for _ in range(length)]
    examples["ability"] = [None for _ in range(length)]
    examples["extra_info"] = [None for _ in range(length)]
    examples["reward_model"] = [{"style": "rule", "ground_truth": None} for _ in range(length)]

    prompts = []
    for system_prompt, user_prompt in zip(system_prompts, user_prompts):
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        messages.append({
            "role": "user",
            "content": user_prompt
        })
        prompts.append(messages)
    examples["prompt"] = prompts

    return examples


if __name__ == '__main__':
    args = parse_arguments()

    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))

    print(f"Loading maga prompt dataset in directory {args.maga_prompt_dataset_path}...")
    dataset = load_dataset("json", data_files=args.maga_prompt_dataset_path)["train"]

    features = list(dataset.features.keys())
    dataset = dataset.map(process, batched=True, batch_size=None, remove_columns=features)

    print(f"Writing prompt dataset in verl format to output path: {args.output_path}...")
    dataset.to_parquet(args.output_path)

    print(f"Done!")
