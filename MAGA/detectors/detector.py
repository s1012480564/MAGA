from detectors.detector_base import Detector
from detectors.models import RoBERTaDetector, SCRNDetector, DetreeDetector, GLTR, DetectGPT, LLMDet, FastDetectGPT, \
    Binoculars, DALD, GECScoreDetector, BERTDetector


def get_detector(detector_name: str, model_paths: list[str] | None) -> Detector | None:
    if "chinese" in detector_name or "zh" in detector_name or "cn" in detector_name:
        return BERTDetector(model_paths[0])
    elif "roberta" in detector_name or detector_name == "radar":
        return RoBERTaDetector(model_paths, detector_name)
    elif detector_name == "scrn":
        return SCRNDetector(model_paths)
    elif detector_name == "detree":
        return DetreeDetector(model_paths)
    elif detector_name == "gltr":
        return GLTR(model_paths[0])
    elif detector_name == "detectgpt":
        return DetectGPT(base_model_name=model_paths[0], mask_filling_model_name=model_paths[1])
    elif detector_name == "llmdet":
        return LLMDet()
    elif detector_name == "fast_detectgpt":
        return FastDetectGPT(model_paths[0], model_paths[1])
    elif detector_name == "binoculars":
        return Binoculars(model_paths[0], model_paths[1])
    elif detector_name == "dald":
        return DALD(model_paths[0], model_paths[1], model_paths[2])
    elif detector_name == "gecscore":
        return GECScoreDetector()
    else:
        raise ValueError("Invalid detector name")
