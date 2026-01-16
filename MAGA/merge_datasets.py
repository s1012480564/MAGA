import os
import argparse
from datasets import load_dataset, concatenate_datasets


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_path", type=str, required=True, help="merge into a jsonl file")
    parser.add_argument("-is", "--input_paths", type=str, default=None, nargs='+',
                        help="path to the datasets to be merged")
    parser.add_argument("-id", "--input_dir", type=str, default=None,
                        help="directory of input files. All first-level files under the directory will be merged. If not None, 'input_paths' argument will not be considered.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))

    dataset_list = []

    print(f"Loading datasets...")
    if args.input_dir:
        for file_name in os.listdir(args.input_dir):
            file_path = os.path.join(args.input_dir, file_name)
            # Skip if it's a directory (only process files) or doesn't have .jsonl extension
            if os.path.isdir(file_path) or not file_name.endswith(".jsonl"):
                continue
            dataset = load_dataset("json", data_files=file_path)["train"]
            dataset_list.append(dataset)

    else:
        dataset_list = [load_dataset("json", data_files=input_path)["train"] for input_path in args.input_paths]

    combined_dataset = concatenate_datasets(dataset_list)

    print(f"Done! Writing combined dataset to output path: {args.output_path}")
    combined_dataset.to_json(args.output_path, orient="records", lines=True)
    print(f"Done!")
