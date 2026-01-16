from detectors.detector_base import Detector
import torch
from transformers import BertTokenizer, BertForSequenceClassification


class BERTDetector(Detector):
    def __init__(self,model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def batch_inference(self, texts: list) -> list:
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        probs = outputs.logits.softmax(dim=-1)
        human_probs, machine_probs = probs[:, 0], probs[:, 1]
        predictions = machine_probs.detach().cpu().numpy().tolist()
        return predictions
