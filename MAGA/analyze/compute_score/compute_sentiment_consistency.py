"""
The model selection alignments HC3, cardiffnlp/twitter-xlm-roberta-base-sentiment is used.

Consistency Calculation:
Cosine Similarity (√): The best choice for consistency calculation.
Manhattan Distance (×): Absolute numerical difference, not suitable.
Pearson Correlation Coefficient (×): Trend consistency between vectors composed of continuous data, not applicable.
"""

import math
import torch
from pprint import pprint
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


def compute_sentiment_polarity(
        texts: list[str],
        model_path: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        batch_size: int = 512
) -> list[list[float]]:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(0)
    model.eval()

    probs = []
    total_samples = len(texts)
    total_batches = math.ceil(total_samples / batch_size)

    with torch.no_grad():
        for batch_idx in tqdm(range(total_batches)):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_samples)
            batch_texts = texts[start_idx:end_idx]

            inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(0)

            outputs = model(**inputs)
            batch_probs = outputs.logits.softmax(dim=-1)

            probs.extend(batch_probs.cpu().numpy().tolist())

    return probs  # negative, neutral, positive


def compute_sentiment_consistency_scores(
        human_texts: list[str],
        machine_texts: list[str],
        model_path: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
) -> dict[str, ...]:
    human_sentiments = compute_sentiment_polarity(human_texts, model_path)
    machine_sentiments = compute_sentiment_polarity(machine_texts, model_path)

    sentiment_consistency_scores = torch.nn.functional.cosine_similarity(
        torch.tensor(human_sentiments).to(0),
        torch.tensor(machine_sentiments).to(0),
        dim=1
    )

    sentiment_consistency_scores = sentiment_consistency_scores.cpu().numpy().tolist()

    results = {
        "human_sentiments": human_sentiments,
        "machine_sentiments": machine_sentiments,
        "sentiment_consistency": sentiment_consistency_scores
    }

    return results


if __name__ == "__main__":
    human_texts = [
        "I'm so sad.",
        "This is human text.",
        "I'm so happy!"
    ]

    machine_texts = [
        "Hello everyone, I am sad today.",
        "This is machine text.",
        "Hello everyone, I am happy today."
    ]

    results = compute_sentiment_consistency_scores(human_texts, machine_texts)

    pprint(results)
