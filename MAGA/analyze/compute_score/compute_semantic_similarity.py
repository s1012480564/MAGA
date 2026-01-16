"""
BERTScore (√)
What's more, obviously, F1 is the preferred indicator for observation.
"""

from pprint import pprint
import bert_score


def compute_bert_scores(reference_texts: list[str], candidate_texts: list[str]) -> dict[str, list[float]]:
    P, R, F1 = bert_score.score(candidate_texts, reference_texts, verbose=True, lang="en", batch_size=512)

    results = {
        "precision": P.numpy().tolist(),
        "recall": R.numpy().tolist(),
        "f1": F1.numpy().tolist()
    }

    return results


if __name__ == "__main__":
    human_texts = [
        "This is human text.",
        "This is another human text."
    ]
    machine_texts = [
        "This is machine text.",
        "This is a new machine text."
    ]

    results = compute_bert_scores(human_texts, machine_texts)

    pprint(results)
