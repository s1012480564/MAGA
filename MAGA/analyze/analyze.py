import logging
import sys
import os
import numpy as np
import pprint
from time import strftime, localtime
import argparse
import pandas as pd
import statistics
from datasets import load_dataset, Dataset
from tqdm import tqdm
from compute_score import compute_content_similarity_scores, compute_bert_scores, compute_lexical_diversity_scores, \
    compute_readability_scores, compute_sentiment_consistency_scores

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
    parser.add_argument("-p", "--model_path", default=None, type=str)
    parser.add_argument("-ah", "--analyze_human", action="store_true")
    parser.add_argument("-ds", "--dimensions", default=["all"], nargs="+", type=str,
                        help="option range [all, content, semantic, diversity, readability, sentiment]")
    return parser.parse_args()


def log_with_pprint_style(data):
    formatted_data = pprint.pformat(data, indent=4)
    logger.info(f"{formatted_data}\n")


def get_avg_min_max_scores(scores: list):
    return statistics.mean(scores), min(scores), max(scores)


def analyze_content_similarity(examples: dict[str, list], dataset_name: str):
    human_texts = examples["text_human"]
    machine_texts = examples["text_machine"]

    rouge_scores_all = {
        "rouge1": {"precision": [], "recall": [], "f1": []},
        "rouge2": {"precision": [], "recall": [], "f1": []},
        "rougeL": {"precision": [], "recall": [], "f1": []}
    }
    rouge_scores, bleu_scores, meteor_scores, avg_content_similarity_scores = [], [], [], []

    for human_text, machine_text in tqdm(zip(human_texts, machine_texts)):
        content_similarity_scores = compute_content_similarity_scores(human_text, machine_text)

        _rouge_score_all = content_similarity_scores["rouge"]

        for rouge_type in ["rouge1", "rouge2", "rougeL"]:
            for metric_type in ["precision", "recall", "f1"]:
                rouge_scores_all[rouge_type][metric_type].append(_rouge_score_all[rouge_type][metric_type])

        _rouge_score = _rouge_score_all["rouge2"]["f1"]
        _bleu_score = content_similarity_scores["bleu"]
        _meteor_score = content_similarity_scores["meteor"]
        avg_score = statistics.mean([_rouge_score, _bleu_score, _meteor_score])

        rouge_scores.append(_rouge_score)
        bleu_scores.append(_bleu_score)
        meteor_scores.append(_meteor_score)
        avg_content_similarity_scores.append(avg_score)

    np.save(f"./analyze_results/data/{dataset_name}/rouge_scores.npy", np.array(rouge_scores))
    np.save(f"./analyze_results/data/{dataset_name}/bleu_scores.npy", np.array(bleu_scores))
    np.save(f"./analyze_results/data/{dataset_name}/meteor_scores.npy", np.array(meteor_scores))
    np.save(f"./analyze_results/data/{dataset_name}/content_similarity_scores.npy",
            np.array(avg_content_similarity_scores))

    avg_content_similarity_score, min_content_similarity_score, max_content_similarity_score = get_avg_min_max_scores(
        avg_content_similarity_scores)
    avg_rouge_score, min_rouge_score, max_rouge_score = get_avg_min_max_scores(rouge_scores)
    avg_bleu_score, min_bleu_score, max_bleu_score = get_avg_min_max_scores(bleu_scores)
    avg_meteor_score, min_meteor_score, max_meteor_score = get_avg_min_max_scores(meteor_scores)

    rouge_summary = {}
    for rouge_type, metrics in rouge_scores_all.items():
        rouge_summary[rouge_type] = {}
        for metric_type, scores in metrics.items():
            avg, min_val, max_val = get_avg_min_max_scores(scores)
            rouge_summary[rouge_type][metric_type] = {
                "avg": avg,
                "min": min_val,
                "max": max_val
            }

    logger.info(f"\n>> Content Similarity Score:\n")

    log_with_pprint_style({
        "content_similarity_score": {
            "avg": avg_content_similarity_score,
            "min": min_content_similarity_score,
            "max": max_content_similarity_score
        }
    })

    log_with_pprint_style({
        "rouge_score": {
            "avg": avg_rouge_score,
            "min": min_rouge_score,
            "max": max_rouge_score
        },
        "bleu_score": {
            "avg": avg_bleu_score,
            "min": min_bleu_score,
            "max": max_bleu_score
        },
        "meteor_score": {
            "avg": avg_meteor_score,
            "min": min_meteor_score,
            "max": max_meteor_score
        }
    })

    log_with_pprint_style({
        "rouge_summary": rouge_summary
    })

    return None


def analyze_semantic_similarity(examples: dict[str, list], dataset_name: str):
    human_texts = examples["text_human"]
    machine_texts = examples["text_machine"]
    bert_scores = compute_bert_scores(human_texts, machine_texts)

    np.save(f"./analyze_results/data/{dataset_name}/bert_scores.npy", np.array(bert_scores["f1"]))

    logger.info(f"\n>> Semantic Similarity Score:\n")

    bert_score_summary = {}
    for metric_type, scores in bert_scores.items():
        avg, min_val, max_val = get_avg_min_max_scores(scores)
        bert_score_summary[metric_type] = {
            "avg": avg,
            "min": min_val,
            "max": max_val
        }

    log_with_pprint_style({
        "bert_score": bert_score_summary
    })


def analyze_lexical_diversity(examples: dict[str, list], dataset_name: str, analyze_human: bool = False):
    human_texts = examples["text_human"]
    machine_texts = examples["text_machine"]

    logger.info(f"\n>> Lexical Diversity Score:\n")

    def _analyze(texts: list[str], is_machine: bool):
        ttr_scores, yules_k_scores, avg_lexical_diversity_scores = [], [], []
        for text in tqdm(texts):
            lexical_diversity_scores = compute_lexical_diversity_scores(text)

            ttr_score = lexical_diversity_scores["ttr"]
            yules_k_score = lexical_diversity_scores["yules_k"]
            # avg_lexical_diversity_score = statistics.mean([ttr_score, yules_k_score])

            ttr_scores.append(ttr_score)
            yules_k_scores.append(yules_k_score)
            # avg_lexical_diversity_scores.append(avg_lexical_diversity_score)

        np.save(f"./analyze_results/data/{dataset_name}/ttr_{"machine" if is_machine else "human"}.npy",
                np.array(ttr_scores))
        np.save(f"./analyze_results/data/{dataset_name}/yules_k_{"machine" if is_machine else "human"}.npy",
                np.array(yules_k_scores))
        # np.save(
        #     f"./analyze_results/data/{dataset_name}/lexical_diversity_scores_{"machine" if is_machine else "human"}.npy",
        #     np.array(avg_lexical_diversity_scores))

        avg_ttr, min_ttr, max_ttr = get_avg_min_max_scores(ttr_scores)
        avg_yules_k, min_yules_k, max_yules_k = get_avg_min_max_scores(yules_k_scores)
        # avg_lexical_diversity_score, min_lexical_diversity_score, max_lexical_diversity_score = get_avg_min_max_scores(
        #     avg_lexical_diversity_scores)

        # log_with_pprint_style({
        #     "lexical_diversity_score": {
        #         "avg": avg_lexical_diversity_score,
        #         "min": min_lexical_diversity_score,
        #         "max": max_lexical_diversity_score
        #     }
        # })

        log_with_pprint_style({
            "ttr": {
                "avg": avg_ttr,
                "min": min_ttr,
                "max": max_ttr
            },
            "yules_k": {
                "avg": avg_yules_k,
                "min": min_yules_k,
                "max": max_yules_k
            }
        })

    logger.info(f">>>> Machine:\n")
    _analyze(machine_texts, is_machine=True)
    if analyze_human:
        logger.info(f">>>> Human:\n")
        _analyze(human_texts, is_machine=False)


def analyze_text_readability(examples: dict[str, list], dataset_name: str, analyze_human: bool = False):
    human_texts = examples["text_human"]
    machine_texts = examples["text_machine"]

    logger.info(f"\n>> Text Readability Score:\n")

    def _analyze(texts: list[str], is_machine: bool):
        flesch_kincaid_scores, smog_scores, dale_chall_scores, avg_readability_scores = [], [], [], []
        for text in tqdm(texts):
            text_readability_scores = compute_readability_scores(text)

            flesch_kincaid_score = text_readability_scores["flesch_kincaid"]
            smog_score = text_readability_scores["smog"]
            dale_chall_score = text_readability_scores["dale_chall"]
            avg_readability_score = statistics.mean([flesch_kincaid_score, smog_score, dale_chall_score])

            flesch_kincaid_scores.append(flesch_kincaid_score)
            smog_scores.append(smog_score)
            dale_chall_scores.append(dale_chall_score)
            avg_readability_scores.append(avg_readability_score)

        np.save(f"./analyze_results/data/{dataset_name}/flesch_kincaid_{"machine" if is_machine else "human"}.npy",
                np.array(flesch_kincaid_scores))
        np.save(f"./analyze_results/data/{dataset_name}/smog_{"machine" if is_machine else "human"}.npy",
                np.array(smog_scores))
        np.save(f"./analyze_results/data/{dataset_name}/dale_chall_{"machine" if is_machine else "human"}.npy",
                np.array(dale_chall_scores))
        np.save(f"./analyze_results/data/{dataset_name}/readability_scores_{"machine" if is_machine else "human"}.npy",
                np.array(avg_readability_scores))

        avg_flesch_kincaid, min_flesch_kincaid, max_flesch_kincaid = get_avg_min_max_scores(flesch_kincaid_scores)
        avg_smog, min_smog, max_smog = get_avg_min_max_scores(smog_scores)
        avg_dale_chall, min_dale_chall, max_dale_chall = get_avg_min_max_scores(dale_chall_scores)
        avg_readability_score, min_readability_score, max_readability_score = get_avg_min_max_scores(
            avg_readability_scores)

        log_with_pprint_style({
            "text_readability_score": {
                "avg": avg_readability_score,
                "min": min_readability_score,
                "max": max_readability_score
            }
        })
        log_with_pprint_style({
            "flesch_kincaid": {
                "avg": avg_flesch_kincaid,
                "min": min_flesch_kincaid,
                "max": max_flesch_kincaid
            },
            "smog": {
                "avg": avg_smog,
                "min": min_smog,
                "max": max_smog
            },
            "dale_chall": {
                "avg": avg_dale_chall,
                "min": min_dale_chall,
                "max": max_dale_chall
            }
        })

    logger.info(f">>>> Machine:\n")
    _analyze(machine_texts, is_machine=True)
    if analyze_human:
        logger.info(f">>>> Human:\n")
        _analyze(human_texts, is_machine=False)


def analyze_sentiment_consistency(examples: dict[str, list], dataset_name: str, model_path: str):
    human_texts = examples["text_human"]
    machine_texts = examples["text_machine"]

    sentiment_consistency_scores_all = compute_sentiment_consistency_scores(human_texts, machine_texts, model_path)

    sentiment_consistency_scores = sentiment_consistency_scores_all["sentiment_consistency"]
    human_sentiments = np.array(sentiment_consistency_scores_all["human_sentiments"])
    machine_sentiments = np.array(sentiment_consistency_scores_all["machine_sentiments"])

    np.save(f"./analyze_results/data/{dataset_name}/sentiment_consistency_scores.npy",
            np.array(sentiment_consistency_scores))

    sentiment_dims = ["negative", "neutral", "positive"]

    for idx, dim in enumerate(sentiment_dims):
        np.save(f"./analyze_results/data/{dataset_name}/human_{dim}_scores.npy", human_sentiments[:, idx])
        np.save(f"./analyze_results/data/{dataset_name}/machine_{dim}_scores.npy", machine_sentiments[:, idx])

    avg_sentiment_consistency_score, min_sentiment_consistency_score, max_sentiment_consistency_score = get_avg_min_max_scores(
        sentiment_consistency_scores)

    def _get_sentiment_stats(sentiment_array):
        stats = {}
        for idx, dim in enumerate(sentiment_dims):
            avg_score, min_score, max_score = get_avg_min_max_scores(sentiment_array[:, idx].tolist())
            stats[dim] = {
                "avg": avg_score,
                "min": min_score,
                "max": max_score
            }
        return stats

    human_sent_stats = _get_sentiment_stats(human_sentiments)
    machine_sent_stats = _get_sentiment_stats(machine_sentiments)

    logger.info(f"\n>> Sentiment Consistency Score:\n")

    log_with_pprint_style({
        "sentiment_consistency_score": {
            "avg": avg_sentiment_consistency_score,
            "min": min_sentiment_consistency_score,
            "max": max_sentiment_consistency_score
        }
    })

    logger.info(f">>>> Machine:\n")
    log_with_pprint_style(machine_sent_stats)

    logger.info(f">>>> Human:\n")
    log_with_pprint_style(human_sent_stats)

    return None


def analyze(examples: dict[str, list], dataset_name: str, dimensions: list[str], analyze_human: bool = None,
            model_path: str = None):
    analyze_content_similarity_or_not = False
    analyze_semantic_similarity_or_not = False
    analyze_lexical_diversity_or_not = False
    analyze_text_readability_or_not = False
    analyze_sentiment_consistency_or_not = False

    if "all" in dimensions:
        analyze_content_similarity_or_not = True
        analyze_semantic_similarity_or_not = True
        analyze_lexical_diversity_or_not = True
        analyze_text_readability_or_not = True
        analyze_sentiment_consistency_or_not = True
    else:
        if "content" in dimensions:
            analyze_content_similarity_or_not = True
        if "semantic" in dimensions:
            analyze_semantic_similarity_or_not = True
        if "diversity" in dimensions:
            analyze_lexical_diversity_or_not = True
        if "readability" in dimensions:
            analyze_text_readability_or_not = True
        if "sentiment" in dimensions:
            analyze_sentiment_consistency_or_not = True

    if analyze_content_similarity_or_not:
        analyze_content_similarity(examples, dataset_name)
    if analyze_semantic_similarity_or_not:
        analyze_semantic_similarity(examples, dataset_name)
    if analyze_lexical_diversity_or_not:
        analyze_lexical_diversity(examples, dataset_name, analyze_human=analyze_human)
    if analyze_text_readability_or_not:
        analyze_text_readability(examples, dataset_name, analyze_human=analyze_human)
    if analyze_sentiment_consistency_or_not:
        analyze_sentiment_consistency(examples, dataset_name, model_path=model_path)


def merge_by_human_source_id(dataset: Dataset, n_sample: int | None = None, seed: int = 42) -> Dataset:
    df = dataset.to_pandas()
    human_df = df[df["label"] == 0]
    machine_df = df[df["label"] == 1]
    merged_df = pd.merge(human_df, machine_df, on="human_source_id", suffixes=("_human", "_machine"))
    if n_sample is not None:
        merged_df = merged_df.sample(n=n_sample, random_state=seed)
    merged_dataset = Dataset.from_pandas(merged_df)
    return merged_dataset


if __name__ == '__main__':
    args = parse_arguments()

    log_file = f"./analyze_results/log/{args.dataset_name}-{strftime("%y%m%d-%H%M", localtime())}.log"

    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    if not os.path.exists(f"./analyze_results/data/{args.dataset_name}/"):
        os.makedirs(os.path.dirname(f"./analyze_results/data/{args.dataset_name}/"), exist_ok=True)

    logger.addHandler(logging.FileHandler(log_file))
    logger.info(f">> n_sample: {args.n_sample}\n")

    print(f"Loading MAGA dataset from {args.maga_dataset_path}...")
    maga_dataset = load_dataset("json", data_files=args.maga_dataset_path)["train"]

    merged_maga_dataset = merge_by_human_source_id(maga_dataset, n_sample=args.n_sample, seed=args.seed)

    merged_maga_dataset.map(
        analyze,
        batched=True,
        batch_size=None,
        fn_kwargs={
            "dataset_name": args.dataset_name,
            "dimensions": args.dimensions,
            "analyze_human": args.analyze_human,
            "model_path": args.model_path
        }
    )
