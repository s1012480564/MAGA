'''
When constructing original prompts,
auxiliary information about which model the prompt is assigned to for generation is also annotated based on MAGA's strategy.
'''

import argparse
import itertools
from uuid import uuid4
from datasets import load_dataset

ORIGINAL_PROMT_TEMPLATE = {
    "Baidu Tieba": "请你为标题为“{title}”的百度贴吧帖子写一条简短的网友回复。请直接给出回复。",
    "Weibo Review": "请你为标题为“{title}”的新浪微博文章写一条简短的网友评论。请直接给出评论。",
    "Rednote Review": "请你为标题为“{title}”的小红书笔记写一条网友评论。请直接给出评论。",
    "CSL": "请你写一段题目为《{title}》的中文核心期刊论文摘要。最好不要以“本文”开头。",
    "Baidu Baike": "请你写一篇标题为“{title}”的百度百科介绍。",
    "Dianping": "请你根据下述关键词写一段大众点评评价。\n关键词：{title}",
    "Douban Review": "请你为电影《{title}》写一段简短的豆瓣影评。请直接给出评论。",
    "Baidu Zhidao": "请你为百度知道提问“{title}”写一段简短的回答。",
    "Zhihu": "请你为知乎提问“{title}”写一段回答。请不要重复问题。",
    "CLTS": "请你写一篇题目为《{title}》的澎湃新闻。请不要重复题目，直接给出正文。"
}

MODEL_LIST = [
    "GPT-4o-mini",
    "Gemini-2.0-flash",
    "DeepSeek-V3",
    "Qwen3-plus",
    "Mistral-Medium",
    "Hunyuan-TurboS",
    "Llama-3.1-8B-Instruct",
    "gemma-3-12b-it",
    "DeepSeek-R1-0528-Qwen3-8B",
    "Qwen3-8B",
    "Ministral-8B-Instruct-2410",
    "Hunyuan-7B-Instruct"
]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, required=True, help="the sampled MAGA human data file path")
    parser.add_argument("-o", "--output_path", type=str, required=True)
    parser.add_argument("--sort-domain", action="store_true",
                        help="according to MAGA's strategy, data from each domain needs to be kept together so that it can be evenly divided among the models for generation. Under the normal MAGA process, data from each domain is already together when merged, so there is no need to re-sort, so this argument can be turned off")
    return parser.parse_args()


def process(examples: dict[str, list]) -> dict[str, list]:
    prompts = []
    for title, domain in zip(examples["title"], examples["domain"]):
        prompt_template = ORIGINAL_PROMT_TEMPLATE[domain]
        prompt = prompt_template.format(title=title)
        prompts.append(prompt)
    examples["prompt_id"] = [str(uuid4()) for _ in range(len(prompts))]
    examples["system_prompt"] = [None for _ in range(len(prompts))]
    examples["user_prompt"] = prompts

    # According to MAGA's strategy, each domain is evenly divided among various models for generation.
    model_names = []
    data_amount_each_domain = [len(list(group)) for _, group in itertools.groupby(examples["domain"])]
    for num in data_amount_each_domain:
        for model_name in MODEL_LIST:
            model_names += [model_name] * (num // len(MODEL_LIST))
    examples["model"] = model_names

    return examples


if __name__ == '__main__':
    args = parse_arguments()

    print(f"Loading sampled MAGA human dataset in directory {args.input_path}...")
    dataset = load_dataset("json", data_files=args.input_path)["train"]

    if args.sort_domain:
        dataset = dataset.sort("domain")

    dataset = dataset.map(process, batched=True, batch_size=None, remove_columns=["text"])

    print(f"Writing original prompt dataset to output path: {args.output_path}...")
    dataset.to_json(args.output_path, orient="records", lines=True)

    print(f"Done!")
