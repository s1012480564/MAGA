'''
For single-sample inference, the loss implemented by the huggingface model is exactly the log ppl.

The calculation rule for the loss in huggingface model involves applying a shift and padding operation to the input labels:
specifically, taking the slice [1:] and appending a padding token at the end (CELoss pad, -100).

It is important to note that during batch inference,
left-padding applied by the tokenizer is included in the batch loss calculation and will not be ignored.

Therefore, if you need to accurately compute the loss (log ppl) for an individual text sample during batch inference,
pay close attention to the starting position of the sample.

Naturally, when calculating the loss for conditional probabilities in a chat scenario with a prompt,
it becomes even more critical to pinpoint the exact starting position of the response.
'''

import logging
import sys
import os
import argparse
import statistics
from datasets import load_dataset
from time import strftime, localtime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

assistant_token_map = {
    "Qwen3-8B": "assistant",
    "Llama-3.1-8B-Instruct": "assistant",
    "glm-4-9b-chat": "<|assistant|>",
}

assistant_gen_idx_delta_map = {
    "Qwen3-8B": 2,
    "Llama-3.1-8B-Instruct": 4,
    "glm-4-9b-chat": 1,
}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_name", required=True, type=str, help="for log")
    parser.add_argument("-m", "--model_name", required=True, type=str)
    parser.add_argument("-i", "--maga_dataset_path", required=True, type=str,
                        help="any MAGA dataset, including MGB/MAGA/MAGA-extra")
    parser.add_argument("-n", "--n_sample", default=None, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("-p", "--model_path", default=None, type=str)
    parser.add_argument("-bs", "--batch_size", default=4, type=int)
    parser.add_argument("-c", "--chat_mode", action="store_true")
    return parser.parse_args()


def compute_ppl(examples: dict[str, list], tokenizer: AutoTokenizer, model: AutoModelForCausalLM, model_name: str,
                chat_mode: bool) -> dict[str, list]:
    log_ppls = []
    texts = examples["text"]
    if not chat_mode:
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False
        )
        with torch.no_grad():
            inputs = inputs.to(model.device)
            input_ids = inputs.input_ids
            outputs = model(
                **inputs,
                labels=input_ids
            )
            logits = outputs.logits
            for input_ids_it, logits_it in zip(input_ids, logits):
                start_idx = torch.where(input_ids_it != tokenizer.pad_token_id)[0][0].item()
                labels_it = input_ids_it[start_idx + 1:]
                loss = F.cross_entropy(logits_it[start_idx:-1, :], labels_it)
                log_ppls.append(loss.item())
    else:
        system_prompts = examples["system_prompt"]
        user_prompts = examples["user_prompt"]
        messages_all = []

        for user_prompt, system_prompt, response in zip(user_prompts, system_prompts, texts):
            messages = []

            if model_name == "Qwen3-8B":
                if system_prompt is None:
                    system_prompt = ""
                system_prompt += "/no_think"
                response = "<think>\n\n</think>\n\n" + response

            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            messages.append({
                "role": "user",
                "content": user_prompt
            })
            messages.append({
                "role": "assistant",
                "content": response
            })
            messages_all.append(messages)

        messages_all_templated = tokenizer.apply_chat_template(
            messages_all,
            tokenize=False,
            add_generation_prompt=False
        )

        inputs = tokenizer(
            messages_all_templated,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False
        )

        with torch.no_grad():
            inputs = inputs.to(model.device)
            input_ids = inputs.input_ids
            outputs = model(
                **inputs,
                labels=input_ids
            )
            logits = outputs.logits
            assistant_token = assistant_token_map[model_name]
            assistant_gen_idx_delta = assistant_gen_idx_delta_map[model_name]
            for input_ids_it, logits_it in zip(input_ids, logits):
                start_idx = torch.where(inputs.input_ids[0] == tokenizer(assistant_token).input_ids[-1])[0][
                                0].item() + assistant_gen_idx_delta
                labels_it = input_ids_it[start_idx + 1:]
                loss = F.cross_entropy(logits_it[start_idx:-1, :], labels_it)
                log_ppls.append(loss.item())

    examples["log_ppl"] = log_ppls
    return examples


if __name__ == '__main__':
    args = parse_arguments()

    log_file = f"./analyze_results/log/ppl/{args.dataset_name}/{args.model_name}-{"chat" if args.chat_mode else "non-chat"}-{strftime("%y%m%d-%H%M", localtime())}.log"

    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger.addHandler(logging.FileHandler(log_file))
    logger.info(f">> n_sample: {args.n_sample}\n")

    print(f"Loading MAGA dataset from {args.maga_dataset_path}...")
    dataset = load_dataset("json", data_files=args.maga_dataset_path)["train"]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        padding_side="left",
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="bfloat16",
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    dataset_machine = dataset.filter(
        lambda examples: [label == 1 for label in examples["label"]],
        batched=True,
        batch_size=None
    )

    if args.n_sample is not None:
        dataset_machine = dataset_machine.shuffle(seed=args.seed)
        dataset_machine = dataset_machine.select(range(args.n_sample))

    dataset_machine = dataset_machine.map(
        compute_ppl,
        batched=True,
        batch_size=args.batch_size,
        fn_kwargs={
            "tokenizer": tokenizer,
            "model": model,
            "model_name": args.model_name,
            "chat_mode": args.chat_mode
        }
    )

    logger.info(f">> mean log ppl:{statistics.mean(list(dataset_machine["log_ppl"]))}")
