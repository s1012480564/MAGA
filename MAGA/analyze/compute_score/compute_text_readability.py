"""
The three most commonly used metrics for text readability are Flesh-Kincaid Reading Ease, SMOG Index, and Dale-Chall Readability Score,
which can basically cover the vast majority of English-language scenarios.

In addition, Flesch-Kincaid also has a grade level metric, and Gunning Fog Index is also relatively commonly used,
but we did not use them here.

Flesch-Kincaid (√): A simple, fast, and intuitive way to calculate whether a text is "easy to understand." Suitable for scenarios such as daily news reports.
SMOG (√): Effectively detects obscure expressions. Suitable for academic papers, instruction manuals, and similar scenarios.
Dale-Chall (√): Based on a basic vocabulary, suitable for children's books and popular science texts.
Flesch-Kincaid Grade: Directly corresponds to school age. Specifically designed for educational scenarios.
Gunning Fog: Assesses obscurity, similar to SMOG.

One minor issue:
For texts with fewer than 3 sentences, it's actually recommended to skip calculating the SMOG index.
This is because the SMOG index wasn't designed for fragmented short sentences.
With only one or two sentences, the SMOG index value will be severely distorted.
Well, actually, it's not a big problem, so we'll ignore it here.
"""

import textstat
from pprint import pprint


def compute_readability_scores(text: str) -> dict[str, float]:
    flesch_kincaid = textstat.flesch_reading_ease(text)
    smog = textstat.smog_index(text)
    dale_chall = textstat.dale_chall_readability_score(text)

    return {
        "flesch_kincaid": flesch_kincaid,
        "smog": smog,
        "dale_chall": dale_chall
    }


if __name__ == "__main__":
    human_text = "Artificial intelligence (AI) refers to the simulation of human intelligence processes by machines, " \
                 "especially computer systems. These processes include learning, reasoning, and self-correction. " \
                 "AI has become an integral part of modern technology, powering everything from virtual assistants " \
                 "to advanced robotics and predictive analytics."
    machine_text = "This is a simple text for testing readability."

    results = compute_readability_scores(human_text)

    pprint(results)

    results = compute_readability_scores(machine_text)

    pprint(results)
