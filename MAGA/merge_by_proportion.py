import argparse
from datasets import load_dataset, concatenate_datasets


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i1", "--input_path_1", type=str, required=True)
    parser.add_argument("-i2", "--input_path_2", type=str, required=True)
    parser.add_argument("-o", "--output_path", type=str, required=True)
    parser.add_argument("-p1", "--input_1_sample_percentage", type=float, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    print(f"Loading input dataset 1: {args.input_path_1}...")
    dataset1 = load_dataset("json", data_files=args.input_path_1)["train"]

    print(f"Loading input dataset 2: {args.input_path_2}...")
    dataset2 = load_dataset("json", data_files=args.input_path_2)["train"]

    percent = args.input_1_sample_percentage
    print(f"Sampling {percent * 100}% from dataset 1...")
    sampled_dataset1 = dataset1.shuffle(seed=42).select(range(int(len(dataset1) * percent)))

    # Collect combinations of (human_source_id, label) from sampled_dataset1
    sampled_pairs = set(zip(sampled_dataset1["human_source_id"], sampled_dataset1["label"]))

    # Filter dataset2: Exclude records where (human_source_id, label) are all in sampled_pairs
    filtered_dataset2 = dataset2.filter(
        lambda x: (x["human_source_id"], x["label"]) not in sampled_pairs
    )

    print(f"Sampled dataset 1 size: {len(sampled_dataset1)}")
    print(f"Filtered dataset 2 size: {len(filtered_dataset2)}")

    combined_dataset = concatenate_datasets([filtered_dataset2, sampled_dataset1])

    print(f"Writing merged dataset to output path: {args.output_path}...")
    combined_dataset.to_json(args.output_path, orient="records", lines=True)

    print(f"Done! Merged dataset size: {len(combined_dataset)}")
