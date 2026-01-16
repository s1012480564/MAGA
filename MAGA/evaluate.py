import os
import argparse
import pandas as pd
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--results_path", type=str, required=True,
                        help="Path to the detection result JSON to evaluate. If label annotated in results, dataset will not be loaded")
    parser.add_argument("-d", "--data_path", type=str, required=True,
                        help="Path to the dataset to evaluate for the results")
    parser.add_argument("-o", "--output_path", type=str, required=True,
                        help="Path to the output JSON to write scores")
    parser.add_argument("-t", "--threshold", type=float, default=0.5,
                        help="Judged as machine text when score >= threshold if machine-like score, else < threshold")
    parser.add_argument("--human-like-score", action="store_true",
                        help="Default is machine-like score, the larger the score, the more machine-like it is. Turn on this argument switches to human score.")
    parser.add_argument(
        "-fpr",
        "--target_fpr",
        nargs="+",
        type=float,
        default=[0.05, 0.03, 0.01],
        help="Target false positive rate to evaluate detectors at",
    )
    parser.add_argument("--on-domain", action="store_true",
                        help="Enable this to display additional results on per domain.")
    return parser.parse_args()


def find_threshold(df: pd.DataFrame, target_fpr: float, human_like_score: bool) -> float:
    """Situation where target_fpr cannot be reached due to rounding due to too few human data is not considered.
    Our metrics contains tnr = 1 - fpr, so you can get real fpr when target fpr is not reached"""
    human_group_df = df[df["label"] == 0]
    n_human = len(human_group_df)
    if human_like_score:
        human_group_df = human_group_df.sort_values(by="score", ascending=True)
    else:
        human_group_df = human_group_df.sort_values(by="score", ascending=False)
    n_human_wrong = int(round(target_fpr * n_human))
    threshold = human_group_df.iloc[n_human_wrong - 1]["score"]
    threshold = threshold - 1e-10 if human_like_score else threshold + 1e-10
    return threshold


def compute_metrics(labels: list[int], scores: list[float], threshold: float, human_like_score: bool) -> tuple[
    float, float, float, float]:
    if human_like_score:
        predictions = [1 if score < threshold else 0 for score in scores]
    else:
        predictions = [1 if score >= threshold else 0 for score in scores]
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return acc, f1, tpr, tnr


def compute_auc(labels: list[int], scores: list[float], human_like_score: bool) -> float:
    if human_like_score:
        reversed_labels = [1 - label for label in labels]
        auc = roc_auc_score(reversed_labels, scores)
    else:
        auc = roc_auc_score(labels, scores)

    return auc


def create_evaluation_result(labels, scores, threshold, fpr, source, human_like_score):
    results = []

    if threshold is not None:
        acc, f1, tpr, tnr = compute_metrics(labels, scores, threshold, human_like_score)
        results.append({
            "scores": {"acc": acc, "f1": f1, "tpr": tpr, "tnr": tnr},
            "threshold": threshold,
            "fpr": fpr,
            "source": source
        })

    return results


def process_group(group_df, args, group_name=None):
    """Process a single group (can be a source group or the entire dataset)"""
    group_results = []
    labels = group_df["label"].tolist()
    scores = group_df["score"].tolist()

    base_results = create_evaluation_result(
        labels, scores, args.threshold, None, group_name, args.human_like_score
    )
    group_results.extend(base_results[:1])

    for target_fpr in args.target_fpr:
        threshold = find_threshold(group_df, target_fpr, args.human_like_score)
        fpr_results = create_evaluation_result(
            labels, scores, threshold, target_fpr, group_name, args.human_like_score
        )
        group_results.extend(fpr_results[:1])

    for score in scores:
        if type(score) is not float:
            print(score)

    base_auc = compute_auc(labels, scores, args.human_like_score)
    group_results.append({
        "scores": {"auc": base_auc},
        "threshold": None,
        "fpr": None,
        "source": group_name
    })

    print(group_results)

    return group_results


def run_evaluation(args, results: Dataset, dataset: Dataset = None) -> list[dict[str, any]]:
    eval_results = []

    if dataset is None:
        merged_df = results.to_pandas()
    else:
        dataset_df = dataset.to_pandas()
        results_df = results.to_pandas()
        merged_df = pd.merge(dataset_df[["id", "label"]], results_df[["id", "score"]],
                             on="id", how="inner")

    overall_results = process_group(merged_df, args, group_name=None)
    eval_results.extend(overall_results)

    if "source" in results.features:
        sources = merged_df["source"].unique()
        for source in sources:
            source_df = merged_df[merged_df["source"] == source].copy()
            source_results = process_group(source_df, args, group_name=source)
            eval_results.extend(source_results)

    if args.on_domain:
        def compute_domain_acc(group_df, domain: str, threshold: float, human_like_score: bool) -> list[dict[str, any]]:
            labels = group_df["label"].tolist()
            scores = group_df["score"].tolist()
            if human_like_score:
                predictions = [1 if score < threshold else 0 for score in scores]
            else:
                predictions = [1 if score >= threshold else 0 for score in scores]
            acc = accuracy_score(labels, predictions)
            group_results = [{
                "scores": {"acc": acc},
                "threshold": threshold,
                "source": domain
            }]
            print(group_results)
            return group_results

        if "domain" in results.features:
            domains = merged_df["domain"].unique()
            for domain in domains:
                domain_df = merged_df[merged_df["domain"] == domain].copy()
                domain_results = compute_domain_acc(domain_df, domain, args.threshold, args.human_like_score)
                eval_results.extend(domain_results)
                # auc = compute_auc(domain_df["label"].tolist(), domain_df["score"].tolist(), args.human_like_score)
                # auc_result = [{
                #     "scores": {"auc": auc},
                #     "threshold": None,
                #     "fpr": None,
                #     "source": domain
                # }]
                # eval_results.extend(auc_result)
        if "sub_source" in results.features:
            domains = merged_df["sub_source"].unique()
            for domain in domains:
                domain_df = merged_df[merged_df["sub_source"] == domain].copy()
                domain_results = compute_domain_acc(domain_df, domain, args.threshold, args.human_like_score)
                eval_results.extend(domain_results)
                # auc = compute_auc(domain_df["label"].tolist(), domain_df["score"].tolist(), args.human_like_score)
                # auc_result = [{
                #     "scores": {"auc": auc},
                #     "threshold": None,
                #     "fpr": None,
                #     "source": domain
                # }]
                # eval_results.extend(auc_result)

    return eval_results


if __name__ == '__main__':
    args = parse_arguments()

    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))

    print(f"Reading detection result at {args.results_path}...")
    results = load_dataset("json", data_files=args.results_path)["train"]

    dataset = None
    if "label" not in results.features:
        print(f"Reading dataset at {args.data_path}...")
        dataset = load_dataset("json", data_files=args.data_path)["train"]

    print(f"Running evaluation...")
    eval_results = run_evaluation(args, results, dataset)

    print(f"Done! Writing evaluation result to output path: {args.output_path}")
    eval_results = Dataset.from_list(eval_results)
    eval_results.to_json(args.output_path, orient="records", lines=True)

    print(f"Done!")
