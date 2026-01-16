import os
import argparse
import pandas as pd
from datasets import load_dataset


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, required=True)
    parser.add_argument("-tr", "--train_output_path", type=str, required=True)
    parser.add_argument("-te", "--test_output_path", type=str, required=True)
    parser.add_argument("-n", "--train_samples_per_domain", type=int, default=6000)
    return parser.parse_args()


def split_dataset_by_domain(dataset, train_samples_per_domain):
    df = dataset.to_pandas()
    train_dfs = []
    test_dfs = []

    for domain, group in df.groupby('domain'):
        # Check if group has enough samples
        if len(group) < train_samples_per_domain:
            raise ValueError(f"Domain '{domain}' has only {len(group)} samples, "
                             f"but {train_samples_per_domain} training samples were requested")

        # Shuffle the group
        shuffled_group = group.sample(frac=1, random_state=42).reset_index(drop=True)
        # Split into train and test
        train_dfs.append(shuffled_group.iloc[:train_samples_per_domain])
        test_dfs.append(shuffled_group.iloc[train_samples_per_domain:])

        print(f"Domain '{domain}': {train_samples_per_domain} training, "
              f"{len(group) - train_samples_per_domain} testing samples")

    # Concatenate all groups
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)

    return train_df, test_df


if __name__ == '__main__':
    args = parse_arguments()

    if not os.path.exists(os.path.dirname(args.train_output_path)):
        os.makedirs(os.path.dirname(args.train_output_path))
    if not os.path.exists(os.path.dirname(args.test_output_path)):
        os.makedirs(os.path.dirname(args.test_output_path))

    print(f"Loading dataset from {args.input_path}...")
    dataset = load_dataset("json", data_files=args.input_path)["train"]
    print(f"Total dataset size: {len(dataset)}")

    print(f"Splitting dataset with {args.train_samples_per_domain} training samples per domain...")
    train_df, test_df = split_dataset_by_domain(dataset, args.train_samples_per_domain)

    print(f"Training set size: {len(train_df)}, Test set size: {len(test_df)}")

    print(f"Writing training dataset to {args.train_output_path}...")
    train_df.to_json(args.train_output_path, orient="records", lines=True)

    print(f"Writing test dataset to {args.test_output_path}...")
    test_df.to_json(args.test_output_path, orient="records", lines=True)

    print(f"Done!")
