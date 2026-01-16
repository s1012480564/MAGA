import os
import argparse
from datasets import load_dataset
from uuid import uuid4

# the original title/text column names in the human source dataset (except "title", "text")
possible_title_column_names = ["summary", "question", "query", "Movie_Name"]
possible_text_column_names = ["body", "abstract", "wiki_intro", "review", "answer", "article", "Comment"]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_dir", type=str, required=True)
    parser.add_argument("-o", "--output_path", type=str, required=True)
    parser.add_argument("-d", "--domain", type=str, required=True)
    parser.add_argument("-t", "--source_file_type", type=str, default="parquet")
    parser.add_argument("-n", "--num", default=7200, type=int)
    return parser.parse_args()


def process(examples: dict[str, list], domain: str) -> dict[str, list]:
    for col_name in possible_title_column_names:
        if col_name in examples:
            examples["title"] = examples[col_name]
            break
    for col_name in possible_text_column_names:
        if col_name in examples:
            examples["text"] = examples[col_name]
            break
    length = len(examples["title"])
    examples["domain"] = [domain] * length
    examples["human_source_id"] = [str(uuid4()) for i in range(length)]
    return examples


if __name__ == '__main__':
    args = parse_arguments()

    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))

    print(f"Loading human source dataset in directory {args.source_dir}...")
    dataset = load_dataset(args.source_file_type, data_dir=args.source_dir)
    dataset = dataset["train"]
    dataset = dataset.shuffle()

    current_column_names = dataset.column_names
    if "title" in current_column_names:
        current_column_names.remove("title")
    if "text" in current_column_names:
        current_column_names.remove("text")

    dataset = dataset.select(range(args.num)).map(
        process, batched=True, batch_size=None,
        remove_columns=current_column_names,
        fn_kwargs={"domain": args.domain}
    )

    print(f"Writing sampled human dataset to output path: {args.output_path}...")
    dataset.to_json(args.output_path, orient="records", lines=True)

    print(f"Done!")
