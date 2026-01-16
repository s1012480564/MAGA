from detectors.detector_base import Detector
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification


class RoBERTaDetector(Detector):
    def __init__(self, model_paths: list[str], detector_name: str = None):
        self.detector_name = detector_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_paths[0])
        self.model = RobertaForSequenceClassification.from_pretrained(model_paths[0]).to(self.device)
        if detector_name:
            if detector_name == "roberta-base-gpt2" or detector_name == "roberta-large-gpt2":
                data = torch.load(model_paths[1], map_location="cpu")
                self.model.load_state_dict(data["model_state_dict"], strict=False)
        self.model.eval()

    def batch_inference(self, texts: list) -> list:
        if self.detector_name == "radar":
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to(
                self.device)
        else:
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        probs = outputs.logits.softmax(dim=-1)
        if self.detector_name == "roberta-base-gpt2" or self.detector_name == "roberta-large-gpt2":
            machine_probs, human_probs = probs[:, 0], probs[:, 1]
        else:
            human_probs, machine_probs = probs[:, 0], probs[:, 1]
        predictions = machine_probs.detach().cpu().numpy().tolist()
        return predictions
