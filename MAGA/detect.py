'''
roberta-base-gpt2: https://openaipublic.azureedge.net/gpt-2/detector-models/v1/detector-base.pt
roberta-large-gpt2: https://openaipublic.azureedge.net/gpt-2/detector-models/v1/detector-large.pt
roberta-base-chatgpt: Hello-SimpleAI/chatgpt-detector-roberta
radar: TrustSafeAI/RADAR-Vicuna-7B
SCRN: CarlanLark/AIGT-detector-mixed-source
DETree: heyongxin233/DETree
roberta-base-chatgpt-chinese: Hello-SimpleAI/chatgpt-detector-roberta-chinese
roberta-MPU-zhv3: yuchuantian/AIGC_detector_zhv3
'''
import os
import argparse
from datasets import load_dataset
from detectors.detector_base import Detector
from detectors.detector import get_detector


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="specify the detector model name you wish to run, e.g. 'roberta-base-gpt2'")
    parser.add_argument("-d", "--data_path", type=str, required=True,
                        help="Specify path to the data file to run detection on. Requires jsonl type.")
    parser.add_argument("-o", "--output_path", type=str, required=True, help="The file path to write the results to")
    parser.add_argument("-p", "--model_paths", type=str, required=True, nargs='+',
                        help="paths to models params, for binoculars e.g. 'tiiuae/falcon-7b tiiuae/falcon-7b-instruct', for roberta-base-gpt2 e.g. 'FacebookAI/roberta-base your_path/detector-base.pt'")
    parser.add_argument("-bs", "--batch_size", default=None, type=int)
    return parser.parse_args()


def process(examples: dict[str, list], detector: Detector) -> dict[str, list]:
    texts = examples["text"]
    examples["score"] = detector.batch_inference(texts)
    return examples


if __name__ == '__main__':
    args = parse_arguments()

    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))

    print(f"Loading dataset with name {args.data_path}...")
    dataset = load_dataset("json", data_files=args.data_path)["train"]

    print(f"Loading detector with name {args.model}...")
    detector = get_detector(args.model, args.model_paths)

    print(f"Running detection...")
    results = dataset.map(process, batched=True, batch_size=args.batch_size,
                          remove_columns=["text"], fn_kwargs={"detector": detector})

    print(f"Done! Writing predictions to output path: {args.output_path}")
    results.to_json(args.output_path, orient="records", lines=True)

    print(f"Done!")
