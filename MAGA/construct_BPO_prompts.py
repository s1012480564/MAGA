'''
When performing batch inference, llama2 has a known issue "RuntimeError: probability tensor contains either inf, nan or element < 0"
(Note: you need to compute on cpu to see the detailed error. On cuda, the error log is rough, only the following effective message:
"_assert_async_cuda_kernel: block: [0,0,0], thread: [0,0,0] Assertion `probability tensor contains either `inf`, `nan` `or element < 0` failed"
"CUDA error: device-side assert triggered")
We found setting dtype bf16 will solve the problem. (llama2's default dtype is fp16)
'''
import os
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

MAGA_ADDITIONAL_INSTRUCTION = {
    "Reddit": " Do not repeat the title.",
    "S2ORC": " It is preferable not to start with \"This paper\".",
    "Trustpilot Reviews": " Do not give it a title.",
    "Amazon Reviews": " Do not give it a title.",
    "Yahoo Answers": " Do not repeat the question.",
    "CC News": " Do not repeat the title.",
    "NPR News": " Do not repeat the title."
}

bpo_prompt_template = "[INST] You are an expert prompt engineer. Please help me improve this prompt to get a more helpful and harmless response:\n{} [/INST]"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--original_prompt_dataset_path", type=str, required=True)
    parser.add_argument("-p", "--model_path", default="THUDM/BPO", type=str)
    parser.add_argument("-o", "--output_path", type=str, required=True)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("-bs", "--batch_size", default=32, type=int)
    return parser.parse_args()


def process(examples: dict[str, list], model, tokenizer) -> dict[str, list]:
    user_prompts = examples["user_prompt"]
    domains = examples["domain"]
    bpo_prompts = []
    for user_prompt, domain in zip(user_prompts, domains):
        if domain in MAGA_ADDITIONAL_INSTRUCTION:
            user_prompt = user_prompt.removesuffix(MAGA_ADDITIONAL_INSTRUCTION[domain])
        bpo_prompt = bpo_prompt_template.format(user_prompt)
        bpo_prompts.append(bpo_prompt)

    model_inputs = tokenizer(bpo_prompts, padding=True, return_tensors="pt").to(model.device)

    outputs = model.generate(**model_inputs, max_new_tokens=1024, do_sample=True, top_p=0.9, temperature=0.6,
                             num_beams=1)

    output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    bpo_responses = [text.split('[/INST]')[1].strip() for text in output_texts]

    for i, response in enumerate(bpo_responses):
        if domains[i] in MAGA_ADDITIONAL_INSTRUCTION:
            bpo_responses[i] = response + MAGA_ADDITIONAL_INSTRUCTION[domains[i]]

    examples["user_prompt"] = bpo_responses
    return examples


if __name__ == '__main__':
    args = parse_arguments()

    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))

    set_seed(args.seed)

    print(f"Loading original prompt dataset in directory {args.original_prompt_dataset_path}...")
    dataset = load_dataset("json", data_files=args.original_prompt_dataset_path)["train"]

    print(f"Loading model from path {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, add_prefix_space=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        # torch_dtype="auto",
        torch_dtype="bfloat16",
        device_map="auto"
    )
    model.config.return_dict = True
    model.eval()

    print(f"Processing...")
    dataset = dataset.map(process, batched=True, batch_size=args.batch_size, fn_kwargs={
        "model": model,
        "tokenizer": tokenizer,
    })

    print(f"Done! Writing role playing prompt dataset to output path: {args.output_path}...")
    dataset.to_json(args.output_path, orient="records", lines=True)

    print(f"Done!")
