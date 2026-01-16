"""
There are many and varied metrics for assessing lexical diversity.
There are no universally accepted, most commonly used metrics.

TTR (Type-Token Ratio) (√): The most basic metric for measuring lexical diversity.

Yule's K (√): Intuitively the most suitable lexical diversity metric for MGT detection area;
it is the core metric for detecting word repetition (one of the typical characteristics of MGT is a high K value).

2-gram vocabulary size: proposed by M4.

"""

from pprint import pprint
import nltk
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


def get_pos(tag):
    return {'J': wordnet.ADJ, 'V': wordnet.VERB, 'N': wordnet.NOUN, 'R': wordnet.ADV}.get(tag[0], wordnet.NOUN)


def preprocess_text(text: str) -> list[str]:
    # Tokenization + Lowercase conversion + Filtering non-alphabetic characters + Stop word removal + Part-of-speech tagging + Lemmatization
    raw_tokens = [
        token.lower()
        for token in word_tokenize(text)
        if token.isalpha() and token.lower() not in STOP_WORDS
    ]

    if not raw_tokens:
        return []

    tagged_tokens = pos_tag(raw_tokens)
    lemmatized_tokens = [
        LEMMATIZER.lemmatize(token, pos=get_pos(tag))
        for token, tag in tagged_tokens
    ]

    return lemmatized_tokens


def compute_ttr(text: str) -> float:
    lemmatized_tokens = preprocess_text(text)

    if not lemmatized_tokens:
        return 0.0

    unique_tokens = len(set(lemmatized_tokens))
    total_tokens = len(lemmatized_tokens)
    ttr = unique_tokens / total_tokens

    return ttr


def compute_yules_k(text: str) -> float:
    lemmatized_tokens = preprocess_text(text)

    if not lemmatized_tokens:
        return 0.0

    word_frequency = Counter(lemmatized_tokens)

    M1 = len(lemmatized_tokens)
    M2 = sum(freq ** 2 for freq in word_frequency.values())

    yules_k = 10000 * (M2 - M1) / (M1 ** 2) if M1 != 0 else 0.0

    return yules_k


def compute_lexical_diversity_scores(text: str) -> dict[str, ...]:
    return {
        "ttr": compute_ttr(text),
        "yules_k": compute_yules_k(text)
    }


if __name__ == "__main__":
    texts = [
        "This is a text.",
        "This was a text. These are sample texts. This is another text."
    ]

    results = compute_lexical_diversity_scores(texts[0])
    pprint(results)

    results = compute_lexical_diversity_scores(texts[1])
    pprint(results)

    print(f"Size of 2-gram vocab: {count_2gram_vocab_size(texts)}")
