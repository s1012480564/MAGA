import argparse
import torch
import random
import tqdm
import json
import numpy as np

from peft import PeftModel
from model import load_model, load_tokenizer
from fast_detect_gpt import get_sampling_discrepancy, get_sampling_discrepancy_analytic
from metrics import get_roc_metrics, get_precision_recall_metrics
from data_builder import load_data

def eval_fastdetect(args):
    scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.eval_dataset_name, args.cache_dir)
    scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
    if args.weight_path is not None:
        scoring_model = PeftModel.from_pretrained(scoring_model, args.weight_path)
    scoring_model.eval()
    reference_model_name = args.reference_model_name
    device = args.device
    cache_dir = args.cache_dir
    # dataset = "WildChat"
    
    reference_tokenizer = load_tokenizer(reference_model_name, args.eval_dataset_name, cache_dir)
    reference_model = load_model(reference_model_name, device, cache_dir)
    reference_model.eval()
    
    
    # load data
    data = load_data(args.eval_dataset_file)
    n_samples = len(data["sampled"])
    # evaluate criterion
    if args.discrepancy_analytic:
        name = "sampling_discrepancy_analytic"
        criterion_fn = get_sampling_discrepancy_analytic
    else:
        name = "sampling_discrepancy"
        criterion_fn = get_sampling_discrepancy

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    results = []
    for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {name} criterion"):
        original_text = data["original"][idx]
        sampled_text = data["sampled"][idx]
        # original text
        tokenized = scoring_tokenizer(original_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(device)
        # tokenized = scoring_tokenizer(original_text, return_tensors="pt", padding="max_length", max_length=200, return_token_type_ids=False).to(args.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = scoring_model(**tokenized).logits[:, :-1]
            
            tokenized = reference_tokenizer(original_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(device)
            # tokenized = reference_tokenizer(original_text, return_tensors="pt", padding="max_length", max_length=200, return_token_type_ids=False).to(args.device)
            assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
            logits_ref = reference_model(**tokenized).logits[:, :-1]
            original_crit = criterion_fn(logits_ref, logits_score, labels)
        # sampled text
        tokenized = scoring_tokenizer(sampled_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(device)
        # tokenized = scoring_tokenizer(sampled_text, return_tensors="pt", padding="max_length", max_length=200, return_token_type_ids=False).to(args.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = scoring_model(**tokenized).logits[:, :-1]
            tokenized = reference_tokenizer(sampled_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(device)
            # tokenized = reference_tokenizer(sampled_text, return_tensors="pt", padding="max_length", max_length=200, return_token_type_ids=False).to(args.device)
            assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
            logits_ref = reference_model(**tokenized).logits[:, :-1]
            sampled_crit = criterion_fn(logits_ref, logits_score, labels)
        # result
        results.append({"original": original_text,
                        "original_crit": original_crit,
                        "sampled": sampled_text,
                        "sampled_crit": sampled_crit})

    # compute prediction scores for real/sampled passages
    predictions = {'real': [x["original_crit"] for x in results],
                   'samples': [x["sampled_crit"] for x in results]}
    print(f"Real mean/std: {np.mean(predictions['real']):.2f}/{np.std(predictions['real']):.2f}, Samples mean/std: {np.mean(predictions['samples']):.2f}/{np.std(predictions['samples']):.2f}")
    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    print(f"Criterion {name}_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
    # results

    dataset_file = args.eval_dataset_file.split("/")[-1]
    output_file = f"{args.output_path}/{dataset_file}"
    results_file = f'{output_file}.{name}.json'
    results = { 'name': f'{name}_threshold',
                'info': {'n_samples': n_samples},
                'predictions': predictions,
                'raw_results': results,
                'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
                'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
                'loss': 1 - pr_auc}
    with open(results_file, 'w') as fout:
        json.dump(results, fout)
        print(f'Results written into {results_file}')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_model_name', type=str, default="llama2-7b")
    parser.add_argument('--scoring_model_name', type=str, default="llama2-7b")
    parser.add_argument('--weight_path', type=str, default="./ckpt/checkpoint-1860")
    parser.add_argument('--target_model_name', type=str, default="ChatGPT")
    parser.add_argument('--output_path', type=str, default="./")
    
    parser.add_argument('--eval_dataset_name', type=str, default="xsum")
    parser.add_argument('--eval_dataset_file', type=str, default="/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_gpt3.5turbo0301/data/xsum_gpt-3.5-turbo-0301")

    parser.add_argument('--discrepancy_analytic', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    eval_fastdetect(args)