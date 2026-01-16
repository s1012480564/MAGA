import os
import sys
import logging
import argparse
from datasets import load_dataset
from time import strftime, localtime
from tqdm import tqdm
from analyze import merge_by_human_source_id

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-name", "--dataset_name", required=True, type=str, help="for log")
    parser.add_argument("-i", "--maga_dataset_path", required=True, type=str,
                        help="any MAGA dataset, including MGB/MAGA/MAGA-extra")
    parser.add_argument("-n", "--n_sample", default=None, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("-ngs", "--ngrams", default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], nargs="+", type=int)
    return parser.parse_args()


def get_ngram_set(texts: list[str], n: int) -> set:
    ngram_set = set()

    for text in tqdm(texts, desc=f"Extracting {n}-grams"):
        tokens = [token.strip().lower() for token in text.split() if token.strip()]

        if len(tokens) >= n:
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i:i + n])
                ngram_set.add(ngram)

    return ngram_set


def analyze_ngram(examples: dict[str, list], n_grams: list[int]):
    machine_texts = examples["text_machine"]
    human_texts = examples["text_human"]

    for n_gram in n_grams:
        machine_ngram_set = get_ngram_set(machine_texts, n_gram)
        human_ngram_set = get_ngram_set(human_texts, n_gram)

        machine_size = len(machine_ngram_set)
        human_size = len(human_ngram_set)

        intersection_set = human_ngram_set.intersection(machine_ngram_set)
        intersection_size = len(intersection_set)

        ratio = round(intersection_size / human_size, 4)

        logger.info(f"\n>>> {n_gram}-gram Vocab Size:\n")
        logger.info(f">>>> Machine: {machine_size}\n")
        logger.info(f">>>> Human: {human_size}")
        logger.info(f">>>> Intersection: {intersection_size}")
        logger.info(f">>>> Intersection / Human Ratio: {ratio}")


if __name__ == '__main__':
    args = parse_arguments()

    log_file = f"./analyze_results/log/n-gram/{args.dataset_name}-{strftime("%y%m%d-%H%M", localtime())}.log"

    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger.addHandler(logging.FileHandler(log_file))

    print(f"Loading MAGA dataset from {args.maga_dataset_path}...")
    maga_dataset = load_dataset("json", data_files=args.maga_dataset_path)["train"]

    merged_maga_dataset = merge_by_human_source_id(maga_dataset, n_sample=args.n_sample, seed=args.seed)

    merged_maga_dataset.map(
        analyze_ngram,
        batched=True,
        batch_size=None,
        fn_kwargs={
            "n_grams": args.ngrams
        }
    )
