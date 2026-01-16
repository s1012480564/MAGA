import os
import copy
import argparse
from uuid import uuid4
from datasets import load_dataset, Dataset
from openai import OpenAI
from tqdm import tqdm

openai_api_key = ""
openai_api_base = ""

model_name_mapping = {
    "DeepSeek-V3": "deepseek-v3",
    "GPT-4o-mini": "gpt-4o-mini-2024-07-18",
    "Gemini-2.0-flash": "models/gemini-2.0-flash",
    "Hunyuan-TurboS": "hunyuan-turbos-20250313",
    "Mistral-Medium": "mistral-medium",
    "Qwen3-plus": "qwen-plus-2025-04-28",
}

feedback_prompts = {
    "Reddit": "Review the tone of the post you just wrote. Does it sound natural and human, or more robotic? If it's not human enough, suggest improvements to make it more conversational and engaging. Only give concise suggestions for improvement. Do not rewrite the post.",
    "S2ORC": "Review the abstract you just wrote. Does it meet the academic rigor of a scientific abstract while sounding natural (avoiding rigid jargon stacking or mechanical statement)? If it's not human enough, suggest improvements to enhance fluency. Only give concise suggestions for improvement. Do not rewrite the abstract.",
    "Wikipedia": "Review the Wikipedia article you just wrote. Does it sound like a human-edited encyclopedia entry? If it's not human enough, suggest improvements to enhance coherence and naturalness. Only give concise suggestions for improvement. Do not rewrite the article.",
    "wikiHow": "Review the wikiHow article you just wrote. Does it have clear, practical steps and sound like a user-friendly guide? If it's not human enough, suggest improvements to enhance usability and approachability. Only give concise suggestions for improvement. Do not rewrite the article.",
    "Trustpilot Reviews": "Review the tone of the review you just wrote. Does it sound natural and human, or more robotic? If it's not human enough, suggest improvements to make it more conversational and engaging. Only give concise suggestions for improvement. Do not rewrite the review.",
    "Amazon Reviews": "Review the tone of the review you just wrote. Does it sound natural and human, or more robotic? If it's not human enough, suggest improvements to make it more conversational and engaging. Only give concise suggestions for improvement. Do not rewrite the review.",
    "Yahoo Answers": "Review the tone of the Yahoo Answers response you just wrote. Does it match the platform's tone (moderately conversational, not too academic)? If it's not human enough, suggest improvements to enhance conversational naturalness. Only give concise suggestions for improvement. Do not rewrite the response.",
    "Natural Questions": "Review the answer you just wrote. Does it accurately solve the question and balance conciseness with completeness while sounding natural? Suggest improvements to enhance naturalness. Only give concise suggestions for improvement. Do not rewrite the answer.",
    "CC News": "Review the news article you just wrote. Does it follow news writing principles and sound like a human-written news piece? If it's not human enough, suggest improvements to enhance objectivity and fluency. Only give concise suggestions for improvement. Do not rewrite the article.",
    "NPR News": "Review the NPR news article you just wrote. Does it match NPR's style? If it's not human enough, suggest improvements to enhance depth and approachability. Only give concise suggestions for improvement. Do not rewrite the article.",
    "Baidu Tieba": "请你回顾你刚撰写的百度贴吧回复，它看起来像是人类写的，还是更像机器生成的？如果它不够像人类写的，请给出改进建议使其更自然且具有对话性。请只给出简洁的改进建议。不要重写回复。",
    "Weibo Review": "请你回顾你刚撰写的新浪微博评论，它看起来像是人类写的，还是更像机器生成的？如果它不够像人类写的，请给出改进建议使其更自然且具有对话性。请只给出简洁的改进建议。不要重写评论。",
    "Rednote Review": "请你回顾你刚撰写的小红书评论，它看起来像是人类写的，还是更像机器生成的？如果它不够像人类写的，请给出改进建议使其更自然且具有对话性。请只给出简洁的改进建议。不要重写评论。",
    "CSL": "请你回顾你刚撰写的中文核心期刊论文摘要，是否兼具学术严谨性和语言流畅性（避免堆砌专业术语或机械陈述）？如果它不够像人类写的，请给出改进建议以增强表达的流畅性。请只给出简洁的改进建议。不要重写摘要。",
    "Baidu Baike": "请你回顾你刚撰写的百度百科介绍，它是否像是人工编写的百科介绍？如果它不够像人类写的，请给出改进建议以增强表达连贯性和自然性。请只给出简洁的改进建议。不要重写百科介绍。",
    "Dianping": "请你回顾你刚撰写的大众点评评价，它看起来是否自然、像人类写的，还是更像机器生成的？如果它不够像人类写的，请给出改进建议使其更自然且具有对话性。请只给出简洁的改进建议。不要重写评价。",
    "Douban Review": "请你回顾你刚撰写的豆瓣影评，它看起来是否自然、像人类写的，还是更像机器生成的？如果它不够像人类写的，请给出改进建议使其更自然且具有对话性。请只给出简洁的改进建议。不要重写评论。",
    "Baidu Zhidao": "请你回顾你刚撰写的百度知道回答，它是否兼顾准确性和实用性，同时读起来自然流畅？请给出改进建议增强表达自然性。请只给出简洁的改进建议。不要重写回答。",
    "Zhihu": "请你回顾你刚撰写的知乎回答，有观点、有依据的同时，它是否符合平台的语气风格（适度对话，不要过于学术化）？如果它不够像人类写的，请给出改进建议以增强表达自然性。请只给出简洁的改进建议。不要重写回答。",
    "CLTS": "请你回顾你刚撰写的澎湃新闻正文，它是否符合新闻写作的客观性和专业性原则，并读起来像是人类写的？如果它不够像人类写的，请给出改进建议以增强客观性和流畅性。请只给出简洁的改进建议。不要重写文章。",
}

refiner_prompts = {
    "Reddit": "Please improve your Reddit post titled \"{title}\" to make it more conversational and engaging. Here are some specific suggestions:\n{feedback}",
    "S2ORC": "Please improve your scientific paper abstract titled \"{title}\" to be more human while maintaining academic rigor. Here are some specific suggestions:\n{feedback}",
    "Wikipedia": "Please improve your Wikipedia article titled \"{title}\" to be more human while adhering to Wikipedia's neutrality standards. Here are some specific suggestions:\n{feedback}",
    "wikiHow": "Please improve your wikiHow article titled \"{title}\" to be more human while keeping steps practical and clear. Here are some specific suggestions:\n{feedback}",
    "Trustpilot Reviews": "Please improve your Trustpilot review titled \"{title}\" to be more human and conversational. Here are some specific suggestions:\n{feedback}.",
    "Amazon Reviews": "Please improve your Amazon review titled \"{title}\" to be more human and conversational. Here are some specific suggestions:\n{feedback}.",
    "Yahoo Answers": "Please improve your Yahoo Answers response to the question titled \"{title}\" to be more human and matching the platform's tone. Here are some specific suggestions:\n{feedback}",
    "Natural Questions": "Please improve your answer to the question \"{title}\" to be more natural while ensuring accuracy. Here are some specific suggestions:\n{feedback}",
    "CC News": "Please improve your news article titled \"{title}\" to be more human while maintaining journalistic objectivity. Here are some specific suggestions:\n{feedback}",
    "NPR News": "Please improve your NPR news article titled \"{title}\" to be more human while keeping NPR's professional and readable style. Here are some specific suggestions:\n{feedback}",
    "Baidu Tieba": "请你优化你为标题为“{title}”的百度贴吧帖子撰写的回复，保持简短并使其更自然且具有对话性。具体改进建议如下：\n{feedback}",
    "Weibo Review": "请你优化你为标题为“{title}”的新浪微博文章撰写的评论，保持简短并使其更自然且具有对话性。具体改进建议如下：\n{feedback}",
    "Rednote Review": "请你优化你为标题为“{title}”的小红书笔记撰写的评论，使其更自然且具有对话性。具体改进建议如下：\n{feedback}",
    "CSL": "请你优化你撰写的题目为《{title}》的中文核心期刊论文摘要，保持学术严谨性的同时使其更像人类写的。具体改进建议如下：\n{feedback}",
    "Baidu Baike": "请你优化你撰写的标题为“{title}”的百度百科介绍，保持百科中立性标准的前提下使其更像人类写的。具体改进建议如下：\n{feedback}",
    "Dianping": "请你优化你基于关键词“{title}”撰写的大众点评评价，使其更具对话性、像人类写的。具体改进建议如下：\n{feedback}",
    "Douban Review": "请你优化你为电影《{title}》撰写的豆瓣影评，保持简短并使其更具对话性、像人类写的。具体改进建议如下：\n{feedback}",
    "Baidu Zhidao": "请你优化你为百度知道提问“{title}”撰写的回答，保持简洁、准确与实用的同时使其更像人类写的。具体改进建议如下：\n{feedback}",
    "Zhihu": "请你优化你为知乎提问“{title}”撰写的回答，使其更像人类写的，并符合平台的风格。具体改进建议如下：\n{feedback}",
    "CLTS": "请你优化你撰写的题目为《{title}》的澎湃新闻，保持新闻客观性的同时使其更像人类写的。具体改进建议如下：\n{feedback}",
}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--prompt_dataset_path", type=str, required=True)
    parser.add_argument("-m", "--model_name", type=str, required=True)
    parser.add_argument("-o", "--output_path", type=str, required=True)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--presence_penalty", default=0, type=float)
    parser.add_argument("--max_tokens", default=1024, type=int)
    parser.add_argument("--filter-model-name", action="store_true",
                        help="according to MAGA's strategy, prompt set's been annotated with which model for generation. Turn on this argument to use only the corresponding prompt")
    parser.add_argument("--self-refine", action="store_true")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))

    dataset = load_dataset("json", data_files=args.prompt_dataset_path)["train"]
    if args.filter_model_name:
        dataset = dataset.filter(
            lambda examples: [model == args.model_name for model in examples["model"]],
            batched=True,
            batch_size=None
        )

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base
    )

    results = []

    print(f"Generating...")
    for input_data in tqdm(dataset):
        messages = []
        system_prompt = input_data["system_prompt"]
        user_prompt = input_data["user_prompt"]

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        messages.append({
            "role": "user",
            "content": user_prompt
        })

        response = client.chat.completions.create(
            model=model_name_mapping[args.model_name],
            messages=messages,
            temperature=args.temperature,
            top_p=args.top_p,
            presence_penalty=args.presence_penalty,
            max_completion_tokens=args.max_tokens
        )

        if args.self_refine:
            domain = input_data["domain"]
            title = input_data["title"]
            feedback_prompt = feedback_prompts[domain]

            answer = response.choices[0].message.content
            messages_answer = copy.deepcopy(messages)
            messages_answer.append({
                "role": "assistant",
                "content": answer
            })

            messages_feedback = copy.deepcopy(messages_answer)
            messages_feedback.append({
                "role": "user",
                "content": feedback_prompt
            })

            response_feedback = client.chat.completions.create(
                model=model_name_mapping[args.model_name],
                messages=messages_feedback,
                temperature=args.temperature,
                top_p=args.top_p,
                presence_penalty=args.presence_penalty,
                max_completion_tokens=args.max_tokens
            )
            feedback = response_feedback.choices[0].message.content

            messages_refine = copy.deepcopy(messages_answer)
            refiner_prompt = refiner_prompts[domain].format(title=title, feedback=feedback)
            messages_refine.append({
                "role": "user",
                "content": refiner_prompt
            })

            response = client.chat.completions.create(
                model=model_name_mapping[args.model_name],
                messages=messages_refine,
                temperature=args.temperature,
                top_p=args.top_p,
                presence_penalty=args.presence_penalty,
                max_completion_tokens=args.max_tokens
            )

        result = {
            "id": str(uuid4()),
            "title": input_data["title"],
            "text": response.choices[0].message.content,
            "domain": input_data["domain"],
            "human_source_id": input_data["human_source_id"],
            "prompt_id": input_data["prompt_id"],
            "system_prompt": input_data["system_prompt"],
            "user_prompt": input_data["user_prompt"],
            "model": args.model_name,
            "label": 1,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": -1,
            "repetition_penalty": -1
        }
        results.append(result)

    result_dataset = Dataset.from_list(results)
    print(f"Writing machine dataset to output path: {args.output_path}...")
    result_dataset.to_json(args.output_path, orient="records", lines=True)
    print(f"Done!")
