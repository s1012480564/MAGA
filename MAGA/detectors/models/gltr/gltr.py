from tqdm import tqdm

from .gltr_model import LM


class GLTR:
    RANK = 10

    def __init__(self, model_path: str) -> None:
        self.model = LM(model_path)

    def batch_inference(self, texts: list) -> list:
        # but in the view of input to neural model, it's actually not a batch.
        predictions = []
        for text in tqdm(texts):
            try:
                results = self.model.check_probabilities(text, topk=1)
            except IndexError as e:
                print(f"Error: {e}")
                predictions.append(1)  # Machine
                continue

            numbers = [item[0] for item in results["real_topk"]]
            numbers = [item <= self.RANK for item in numbers]

            predictions.append(sum(numbers) / len(numbers))

        return predictions

    @classmethod
    def set_param(cls, rank: int, prob: float) -> None:
        cls.RANK = rank
