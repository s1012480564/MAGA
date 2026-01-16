import os
import argparse
from datasets import load_dataset


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dataset_path", type=str, required=True)
    parser.add_argument("-o", "--output_path", default=None, type=str)
    return parser.parse_args()


def process(examples: dict[str, list]) -> dict[str, list]:
    fixed_titles = [f"How to {title}" for title in examples["title"]]
    examples["title"] = fixed_titles
    return examples


if __name__ == '__main__':
    args = parse_arguments()

    if args.output_path is None:
        args.output_path = args.input_dataset_path
    elif not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))

    print(f"Loading dataset from {args.input_dataset_path}...")
    dataset = load_dataset("json", data_files=args.input_dataset_path)["train"]

    dataset = dataset.map(
        process,
        batched=True,
        batch_size=None
    )

    print(f"Saving fixed dataset to {args.output_path}...")
    dataset.to_json(args.output_path, orient="records", lines=True)

    print(f"Done!")
