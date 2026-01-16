import os
import argparse
import random
from datasets import load_dataset


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--original_prompt_dataset_path", type=str, required=True,
                        help="Pass in dataset in the form processed by construct_original_prompts.py. It's OK to have it processed further.")
    parser.add_argument("-r", "--role_playing_prompt_path", type=str, required=True)
    parser.add_argument("-o", "--output_path", type=str, required=True)
    return parser.parse_args()


def process(examples: dict[str, list], role_prompts: list[str]) -> dict[str, list]:
    length = len(examples["system_prompt"])
    system_prompts = [random.choice(role_prompts) for _ in range(length)]
    examples["system_prompt"] = system_prompts
    return examples


if __name__ == '__main__':
    args = parse_arguments()

    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))

    print(f"Loading original prompt dataset in directory {args.original_prompt_dataset_path}...")
    dataset = load_dataset("json", data_files=args.original_prompt_dataset_path)["train"]

    print(f"Loading role playing prompt dataset from {args.role_playing_prompt_path}...")
    role_dataset = load_dataset("json", data_files=args.role_playing_prompt_path)["train"]
    role_prompts = role_dataset["system_prompt"]

    dataset = dataset.map(
        process,
        batched=True,
        batch_size=None,
        fn_kwargs={"role_prompts": role_prompts}
    )

    print(f"Writing role playing prompt dataset to output path: {args.output_path}...")
    dataset.to_json(args.output_path, orient="records", lines=True)

    print(f"Done!")
