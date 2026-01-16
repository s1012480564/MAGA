from detectors.detector_base import Detector
import torch
from transformers import RobertaConfig, RobertaTokenizer
from .utils.scrn_model import SCRNModel


class SCRNDetector(Detector):
    def __init__(self, model_paths: list[str]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = RobertaConfig.from_pretrained(model_paths[0])
        self.tokenizer = RobertaTokenizer.from_pretrained(model_paths[0])
        self.model = SCRNModel(model_paths[0], self.config)
        state_dicts = torch.load(model_paths[1])
        state_dicts.pop("bert.embeddings.position_ids")  # fix version bug
        self.model.load_state_dict(state_dicts)
        self.model.to(self.device)
        self.model.eval()

    def batch_inference(self, texts: list) -> list:
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        probs = outputs.logits.softmax(dim=-1)
        human_probs, machine_probs = probs[:, 0], probs[:, 1]
        predictions = machine_probs.detach().cpu().numpy().tolist()
        return predictions
