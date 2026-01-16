class Detector:
    """Shared interface for all detectors"""

    def batch_inference(self, texts: list) -> list:
        """Takes in a list of texts and outputs a list of scores from 0 to 1 with
        0 indicating likely human-written, and 1 indicating likely machine-generated."""
        pass
