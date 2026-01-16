import os
import re
import argparse
from datasets import load_dataset


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dataset_path", type=str, required=True)
    parser.add_argument("-o", "--output_dataset_path", type=str, required=True)
    return parser.parse_args()


def process(examples: dict[str, list]) -> dict[str, list]:
    processed_texts = []

    for i in range(len(examples["domain"])):
        domain = examples["domain"][i]
        text = examples["text"][i]
        text = text.lstrip()

        if domain == "S2ORC":
            processed_text = text.lstrip(" \n:*#")
            processed_text = processed_text.removeprefix("Abstract")
            processed_text = processed_text.removeprefix("abstract")
            processed_text = processed_text.lstrip(" \n:*#")

        elif domain == "Wikipedia":
            label = examples["label"][i]
            title = examples["title"][i]

            if label == 0:
                processed_text = text.lstrip()
                processed_text = processed_text.removeprefix(title)
                processed_text = processed_text.lstrip()
            else:  # label == 1
                if text.startswith("*") or text.startswith("#") or text.startswith(title + ":") or text.startswith(
                        title + " :"):
                    processed_text = text.lstrip(" \n:*#")
                    processed_text = processed_text.removeprefix(title)
                    processed_text = processed_text.lstrip(" \n:*#")
                else:
                    processed_text = text

            processed_text = processed_text.replace("- ", "")
            processed_text = processed_text.replace("**", "")
            processed_text = processed_text.replace("# ", "").replace("## ", "").replace("### ", "")

            processed_text = re.sub(r'\n+', '\n', processed_text)

        elif domain == "wikiHow":
            label = examples["label"][i]

            if label == 1:
                # processed_text = text.lstrip(" \n:*#")
                # title = examples["title"][i]
                # processed_text = processed_text.removeprefix(title)
                # processed_text = processed_text.lstrip(" \n:*#")
                #
                # newline_index = processed_text.find("\n\n")
                # if newline_index != -1:
                #     processed_text = processed_text[:newline_index]

                # We eventually found that more than half of the generations in wikiHow are actually irregular and do not conform to the above rules.
                # We found that the longest wikiHow title in our dataset does not exceed 108 characters.
                # We found that generations in wikiHow always have "\n\n" delimiters, and the first section of main text is long enough.
                # The probability of this not happening accidentally is far less than 0.001%.

                text_splits = text.split("\n\n")
                has_long_split = False
                for i in range(1, len(text_splits)):
                    text_split = text_splits[i]
                    if len(text_split) >= 128:
                        processed_text = text_split
                        has_long_split = True
                        break
                if not has_long_split:  # The probability of this accident is far less than 0.001%.
                    processed_text = text
            else:
                processed_text = text

        elif domain == "CSL":
            label = examples["label"][i]
            title = examples["title"][i]

            if label == 1:
                processed_text = text.removeprefix(f"《{title}》").removeprefix("摘要：").lstrip()
            else:
                processed_text = text

        elif domain == "Baidu Baike":
            label = examples["label"][i]
            title = examples["title"][i]

            if label == 1:
                processed_text = text.removeprefix(f"标题：{title}").removeprefix(f"**{title}**").removeprefix(
                    f"【{title}】").removeprefix(f"# {title}").lstrip()

                processed_text = processed_text.replace("- ", "")
                processed_text = processed_text.replace("**", "")
                processed_text = processed_text.replace("# ", "").replace("## ", "").replace("### ", "")
                processed_text = processed_text.replace("【", "").replace("】", "")

                chinese_numbers = ["一", "二", "三", "四", "五", "六", "七", "八", "九"]
                for num in chinese_numbers:
                    processed_text = processed_text.replace(f"{num}、", "")

                for j in range(1, 10):
                    processed_text = processed_text.replace(f"{j}. ", "")

                processed_text = re.sub(r'\n+', '\n', processed_text)
                processed_text = processed_text.lstrip()
            else:
                processed_text = text

        elif domain == "Zhihu":
            label = examples["label"][i]

            if label == 1:
                processed_text = re.sub(r'\n+', ' ', text)
            else:
                processed_text = text

        elif domain == "CLTS":
            label = examples["label"][i]

            if label == 1:
                processed_text = text.replace('\n', '')
            else:
                processed_text = text

        else:
            processed_text = text

        processed_text = processed_text.lstrip()
        processed_texts.append(processed_text)

    examples["text"] = processed_texts
    return examples


if __name__ == '__main__':
    args = parse_arguments()

    output_dir = os.path.dirname(args.output_dataset_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading dataset with name {args.input_dataset_path}...")
    dataset = load_dataset("json", data_files=args.input_dataset_path)["train"]

    print("Processing...")
    dataset = dataset.map(
        process,
        batched=True,
        batch_size=None
    )

    print(f"Done! Writing processed dataset to output path: {args.output_dataset_path}...")
    dataset.to_json(args.output_dataset_path, orient="records", lines=True)

    print(f"Done!")
