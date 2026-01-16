import os
import argparse
from datasets import load_dataset, concatenate_datasets


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ih", "--human_dataset_path", type=str, required=True)
    parser.add_argument("-im", "--machine_dataset_path", type=str, required=True)
    parser.add_argument("-o", "--output_path", type=str, required=True)
    return parser.parse_args()


def process(examples: dict[str, list]) -> dict[str, list]:
    length = len(examples["domain"])
    examples["id"] = examples["human_source_id"]
    examples["model"] = ["human"] * length
    examples["label"] = [0] * length
    for key in ["prompt_id", "system_prompt", "user_prompt", "temperature", "top_p", "top_k", "repetition_penalty"]:
        if key not in examples:
            examples[key] = [None] * length
    return examples


if __name__ == '__main__':
    args = parse_arguments()

    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))

    print(f"Loading human dataset with name {args.human_dataset_path}...")
    human_dataset = load_dataset("json", data_files=args.human_dataset_path)["train"]

    print(f"Loading machine dataset with name {args.machine_dataset_path}...")
    machine_dataset = load_dataset("json", data_files=args.machine_dataset_path)["train"]

    print(f"Processing...")
    human_dataset = human_dataset.map(process, batched=True, batch_size=None)
    combined_dataset = concatenate_datasets([machine_dataset, human_dataset])

    print(f"Done! Writing combined dataset to output path: {args.output_path}")
    combined_dataset.to_json(args.output_path, orient="records", lines=True)

    print(f"Done!")
