from pathlib import Path
from .detree.inference import Detector


class DetreeDetector:
    def __init__(self, model_paths: list[str]):
        # model_paths[1] is not model_path actually. Provide detree's embeddings path for it
        self.detector = Detector(
            database_path=Path(model_paths[1]),
            model_name_or_path=model_paths[0],
            batch_size=128
        )

    def batch_inference(self, texts: list) -> list:
        predictions = self.detector.predict(texts)
        print(predictions[0].probability_ai)
        predictions = [pred.probability_ai for pred in predictions]
        return predictions
