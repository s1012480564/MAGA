import os
import argparse
import random
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from datasets import load_dataset, ClassLabel, Dataset
from tqdm import tqdm
from time import strftime, localtime


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--maga_dataset_path", required=True, type=str,
                        help="any MAGA dataset, including MGB/MAGA/MAGA-extra")
    parser.add_argument("-n", "--n_sample_per_class", default=None, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("-p", "--model_path", default=None, type=str)
    parser.add_argument("-bs", "--batch_size", default=512, type=int)
    parser.add_argument("-ah", "--analyze_human", action="store_true")
    return parser.parse_args()


def get_embeddings(texts: list[str], model_name_or_path: str = "BAAI/bge-large-en-v1.5", batch_size: int = 512):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    model = AutoModel.from_pretrained(model_name_or_path).to(0)
    model.eval()

    embeddings = None

    for i in tqdm(range(0, len(texts), batch_size), desc="generating embeddings: "):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt').to(0)

        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()

            embeddings = batch_embeddings if embeddings is None else np.concatenate(
                [embeddings, batch_embeddings], axis=0
            )

    return embeddings


def draw_tsne(embeddings: np.ndarray, unique_labels: list[str], n_sample_per_class: int,
              title_name: str = None) -> np.ndarray:
    print("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)
    print("t-SNE dimensionality reduction completed!")

    plt.figure(figsize=(12, 10))

    class_label = ClassLabel(names=unique_labels)
    encoded_labels = []
    for label_id in class_label.str2int(unique_labels):
        encoded_labels += [label_id] * n_sample_per_class

    scatter = plt.scatter(
        x=tsne_results[:, 0],
        y=tsne_results[:, 1],
        s=30,
        c=encoded_labels,
        alpha=0.7
    )

    handles, _ = scatter.legend_elements()
    plt.legend(handles, unique_labels)

    if title_name:
        plt.title(title_name, fontsize=15)

    # plt.show()
    # plt.savefig("temp.jpg", dpi=300, bbox_inches='tight')
    plt.savefig(f"./analyze_results/figures/tsne/{strftime("%y%m%d-%H%M", localtime())}.pdf", format="pdf")

    return tsne_results


def filter_datasets(dataset: Dataset, unique_labels: list[str], class_type: str, n_sample_per_class: int) -> \
        list[Dataset]:
    '''
    :param dataset:
    :param unique_labels:
    :param class_type: "domain" or "model"
    :param n_sample_per_class:
    :return:
    '''
    datasets_on_class = []
    for label in unique_labels:
        if class_type == "domain":
            dataset_on_class = dataset.filter(
                lambda examples: [label == domain for domain in examples["domain"]],
                batched=True,
                batch_size=None
            )
        elif class_type == "model":
            dataset_on_class = dataset.filter(
                lambda examples: [label == model for model in examples["model"]],
                batched=True,
                batch_size=None
            )

        dataset_on_class = dataset_on_class.select(random.sample(range(len(dataset_on_class)), n_sample_per_class))
        datasets_on_class.append(dataset_on_class)

    return datasets_on_class


def draw_tsne_from_datasets(datasets_on_class: list[Dataset], unique_labels: list[str], n_sample_per_class: int,
                            model_name_or_path: str = "BAAI/bge-large-en-v1.5", batch_size: int = 512):
    texts = []
    for dataset_on_class in datasets_on_class:
        texts.extend(list(dataset_on_class["text"]))

    embeddings = get_embeddings(
        texts=texts,
        model_name_or_path=model_name_or_path,
        batch_size=batch_size,
    )

    tsne_results = draw_tsne(
        embeddings=embeddings,
        unique_labels=unique_labels,
        n_sample_per_class=n_sample_per_class
    )

    np.save(f"./analyze_results/data/tsne/{strftime("%y%m%d-%H%M", localtime())}.npy", tsne_results)


if __name__ == '__main__':
    args = parse_arguments()

    if not os.path.exists(os.path.dirname("./analyze_results/figures/tsne/")):
        os.makedirs(os.path.dirname("./analyze_results/figures/tsne/"), exist_ok=True)
    if not os.path.exists(f"./analyze_results/data/tsne/"):
        os.makedirs(os.path.dirname(f"./analyze_results/data/tsne/"), exist_ok=True)

    print(f"Loading MAGA dataset from {args.maga_dataset_path}...")
    dataset = load_dataset("json", data_files=args.maga_dataset_path)["train"]

    if args.n_sample_per_class is not None:
        dataset = dataset.shuffle(seed=args.seed)

    dataset_machine = dataset.filter(
        lambda examples: [label == 1 for label in examples["label"]],
        batched=True,
        batch_size=None
    )

    domains = list(set(dataset_machine["domain"]))
    models = list(set(dataset_machine["model"]))

    n_sample_per_class = args.n_sample_per_class
    if n_sample_per_class is None:
        n_sample_per_class = len(dataset_machine) / len(domains)
    datasets_on_class = filter_datasets(
        dataset=dataset_machine,
        unique_labels=domains,
        class_type="domain",
        n_sample_per_class=n_sample_per_class
    )
    draw_tsne_from_datasets(
        datasets_on_class=datasets_on_class,
        unique_labels=domains,
        n_sample_per_class=n_sample_per_class,
        model_name_or_path=args.model_path,
        batch_size=args.batch_size
    )

    # models = ["Qwen3-8B", "DeepSeek-R1-0528-Qwen3-8B", "Llama-3.1-8B-Instruct", "gemma-3-12b-it"]
    n_sample_per_class = args.n_sample_per_class
    if n_sample_per_class is None:
        n_sample_per_class = len(dataset_machine) / len(models)
    datasets_on_class = filter_datasets(
        dataset=dataset_machine,
        unique_labels=models,
        class_type="model",
        n_sample_per_class=n_sample_per_class
    )
    draw_tsne_from_datasets(
        datasets_on_class=datasets_on_class,
        unique_labels=models,
        n_sample_per_class=n_sample_per_class,
        model_name_or_path=args.model_path,
        batch_size=args.batch_size
    )

    if args.analyze_human:
        dataset_human = dataset.filter(
            lambda examples: [label == 0 for label in examples["label"]],
            batched=True,
            batch_size=None
        )
        # domains = ["Wikipedia", "wikiHow", "Trustpilot Reviews", "Amazon Reviews"]
        n_sample_per_class = args.n_sample_per_class
        if n_sample_per_class is None:
            n_sample_per_class = len(dataset_human) / len(domains)
        datasets_on_class = filter_datasets(
            dataset=dataset_human,
            unique_labels=domains,
            class_type="domain",
            n_sample_per_class=n_sample_per_class
        )
        draw_tsne_from_datasets(
            datasets_on_class=datasets_on_class,
            unique_labels=domains,
            n_sample_per_class=n_sample_per_class,
            model_name_or_path=args.model_path,
            batch_size=args.batch_size
        )
