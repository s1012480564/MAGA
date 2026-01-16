"""
ROUGE (√): Abstract/Translation
BLEU (√): Translation
METEOR (√): Multi-dimensional matching, filling the gaps in synonym matching
CHRF (×): Character-level matching, translation of less commonly spoken languages/short texts
CIDEr (×): Image description

ROUGE-1: Unigram-based matching. Measures basic semantic coverage, suitable for short texts (e.g., short sentence summaries)
ROUGE-2: Bigram-based matching. Measures phrase-level semantic coherence, suitable for long texts (e.g., long document summaries)
ROUGE-L: Longest Common Subsequence (LCS)-based matching. Measures word order-level semantic matching,
suitable for tasks requiring word order preservation (e.g., machine translation).

Text summarization generation, priority for reference: R > F1 > P.
The core requirement is "covering key information in the original text."
Missing information is more detrimental than adding redundant sentences (e.g., omitting key events in a news summary).

Machine translation/question answering, priority for reference: P > F1 > R.
The core requirement is "outputting accurate information."
Mistranslations or irrelevant answers negatively impact the user experience more than under-translation/under-answering
 (e.g., incorrect translation of key terms).

General content generation, prioritize F1.
The core requirement is to avoid omitting key information and generating irrelevant content
(e.g., copywriting generation, report abbreviations), balancing coverage and accuracy.

Therefore, we take ROUGE-2-F1 as the core reference here.

"""

from pprint import pprint
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

nltk.download('punkt')  # Required for word segmentation
nltk.download('wordnet')  # Required for METEOR synonym matching
nltk.download('punkt_tab')  # Required for wordnet
nltk.download('omw-1.4')  # Multilingual WordNet extension

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def compute_rouge_score(reference_text: str, candidate_text: str) -> dict[str, dict[str, float]]:
    scores = scorer.score(reference_text, candidate_text)
    result = {}
    for score_name, score_values in scores.items():
        result[score_name] = {
            "precision": score_values.precision,
            "recall": score_values.recall,
            "f1": score_values.fmeasure
        }
    return result


def compute_bleu_score(reference_text: str, candidate_text: str) -> float:
    # Word segmentation and unified conversion to lowercase (eliminating the influence of capitalization)
    ref_tokens = word_tokenize(reference_text.lower())
    cand_tokens = word_tokenize(candidate_text.lower())
    references = [ref_tokens]
    # Smoothing function. Prevents zero scores. No external corpus, mostly long texts, so method3 is selected.
    # (method 4 is also reasonable, the preferred and most commonly used method in NLG.)
    smooth_fn = SmoothingFunction().method4
    # default average of 1-4 grams.
    bleu = sentence_bleu(
        references=references,
        hypothesis=cand_tokens,
        smoothing_function=smooth_fn
    )
    return bleu


def compute_meteor_score(reference_text: str, candidate_text: str) -> float:
    # Word segmentation and unified conversion to lowercase (eliminating the influence of capitalization)
    ref_tokens = word_tokenize(reference_text.lower())
    cand_tokens = word_tokenize(candidate_text.lower())
    meteor = meteor_score([ref_tokens], cand_tokens)
    return meteor


def compute_content_similarity_scores(reference_text: str, candidate_text: str) -> dict[str, ...]:
    return {
        "rouge": compute_rouge_score(reference_text, candidate_text),
        "bleu": compute_bleu_score(reference_text, candidate_text),
        "meteor": compute_meteor_score(reference_text, candidate_text)
    }


if __name__ == "__main__":
    human_text = "This is human text."
    machine_text = "This is machine text."

    results = compute_content_similarity_scores(human_text, machine_text)

    pprint(results)
