import os
import torch
import numpy as np
from tqdm import tqdm

from .scripts.model import load_model, load_tokenizer
from .scripts.fast_detect_gpt import get_sampling_discrepancy_analytic, get_sampling_discrepancy
from peft import PeftModel


class DALD:
    def __init__(self,
                 scoring_model_name="llama2-7b",
                 reference_model_name="llama2-7b",
                 weight_path=None,
                 discrepancy_analytic=True,
                 device="cuda",
                 cache_dir="./cache"
                 ):
        self.scoring_model_name = scoring_model_name
        self.reference_model_name = reference_model_name
        self.discrepancy_analytic = discrepancy_analytic
        self.device = device
        self.cache_dir = cache_dir
        self.weight_path = weight_path

        # Load models and tokenizers
        self.scoring_tokenizer = load_tokenizer(self.scoring_model_name, "xsum", self.cache_dir)
        self.scoring_model = load_model(self.scoring_model_name, "cpu", self.cache_dir)  # Load on CPU first

        # Load PEFT weights if provided
        if self.weight_path is not None:
            self.scoring_model = PeftModel.from_pretrained(self.scoring_model, self.weight_path)

        self.scoring_model.eval()
        self.scoring_model.to(self.device)  # Move to device after PEFT loading

        self.reference_tokenizer = load_tokenizer(self.reference_model_name, "xsum", self.cache_dir)
        self.reference_model = load_model(self.reference_model_name, self.device, self.cache_dir)
        self.reference_model.eval()

        # Set criterion function
        if self.discrepancy_analytic:
            self.criterion_fn = get_sampling_discrepancy_analytic
        else:
            self.criterion_fn = get_sampling_discrepancy

    def batch_inference(self, texts: list) -> list:
        # but in the view of input to neural model, it's actually not a batch.
        predictions = []
        for text in tqdm(texts, desc="Running DALD inference"):
            # Tokenize text with scoring tokenizer
            tokenized = self.scoring_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
                return_token_type_ids=False
            ).to(self.device)

            labels = tokenized.input_ids[:, 1:]

            with torch.no_grad():
                logits_score = self.scoring_model(**tokenized).logits[:, :-1]

                # Tokenize with reference tokenizer
                tokenized_ref = self.reference_tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048,
                    return_token_type_ids=False
                ).to(self.device)

                assert torch.all(tokenized_ref.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = self.reference_model(**tokenized_ref).logits[:, :-1]

                # Calculate discrepancy score
                score = self.criterion_fn(logits_ref, logits_score, labels)
                predictions.append(score)

        return predictions

    def interactive(self):
        print("Welcome to DALD Interactive Mode!")
        print("Enter text to analyze, or 'quit' to exit.")

        while True:
            text = input("\nEnter text: ")
            if text.lower() == 'quit':
                break

            score = self.batch_inference([text])[0]
            print(f"DALD Score: {score:.4f}")
