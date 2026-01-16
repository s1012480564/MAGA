import os
import sys
import logging
import argparse
from datasets import load_dataset
from time import strftime, localtime
from tqdm import tqdm
from analyze import merge_by_human_source_id, log_with_pprint_style

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-name", "--dataset_name", required=True, type=str, help="for log")
    parser.add_argument("-i", "--maga_dataset_path", required=True, type=str,
                        help="any MAGA dataset, including MGB/MAGA/MAGA-extra")
    parser.add_argument("-ah", "--analyze_human", action="store_true")
    return parser.parse_args()


def count_2gram_vocab_size(texts: list[str]) -> int:
    vocab_2gram = set()

    for text in tqdm(texts):
        tokens = [token.strip().lower() for token in text.split() if token.strip()]

        if len(tokens) >= 2:
            for i in range(len(tokens) - 1):
                bigram = (tokens[i], tokens[i + 1])
                vocab_2gram.add(bigram)

    return len(vocab_2gram)


def analyze_2_gram_vocab_size(examples: dict[str, list], analyze_human: bool = None):
    human_texts = examples["text_human"]
    machine_texts = examples["text_machine"]
    logger.info(f">>>> Machine:\n")
    bigram_vocab_size_machine = count_2gram_vocab_size(machine_texts)
    log_with_pprint_style({
        "2gram_vocab_size": bigram_vocab_size_machine
    })
    if analyze_human:
        logger.info(f">>>> Human:\n")
        bigram_vocab_size_human = count_2gram_vocab_size(human_texts)
        log_with_pprint_style({
            "2gram_vocab_size": bigram_vocab_size_human
        })


if __name__ == '__main__':
    args = parse_arguments()

    log_file = f"./analyze_results/log/2-gram/{args.dataset_name}-{strftime("%y%m%d-%H%M", localtime())}.log"

    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger.addHandler(logging.FileHandler(log_file))

    print(f"Loading MAGA dataset from {args.maga_dataset_path}...")
    maga_dataset = load_dataset("json", data_files=args.maga_dataset_path)["train"]

    logger.info(f"\n>> Total:\n")

    merged_maga_dataset = merge_by_human_source_id(maga_dataset)

    merged_maga_dataset.map(
        analyze_2_gram_vocab_size,
        batched=True,
        batch_size=None,
        fn_kwargs={
            "analyze_human": args.analyze_human
        }
    )

    for target_model_name in merged_maga_dataset.unique("model_machine"):
        logger.info(f">>> {target_model_name}:\n")

        merged_maga_dataset_on_model = merged_maga_dataset.filter(
            lambda examples: [model_name == target_model_name for model_name in examples["model_machine"]],
            batched=True,
            batch_size=None
        )

        merged_maga_dataset_on_model.map(
            analyze_2_gram_vocab_size,
            batched=True,
            batch_size=None,
            fn_kwargs={
                "analyze_human": args.analyze_human
            }
        )
