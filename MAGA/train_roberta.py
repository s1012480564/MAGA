'''
If you want to train bert-base-chinese, please change all the roberta class to bert or auto.
'''
from sklearn import metrics
import argparse
from datasets import load_dataset
from transformers import RobertaConfig, RobertaTokenizerFast, RobertaForSequenceClassification, TrainingArguments, \
    Trainer, EvalPrediction, PreTrainedModel
from torch.nn.init import kaiming_uniform_
from functools import partial
import torch.nn as nn


def preprocess(examples: dict[str, list], tokenizer: RobertaTokenizerFast) -> dict[str, list]:
    examples["labels"] = examples["label"]
    encoded_texts = tokenizer(examples["text"], padding=True, truncation=True)
    examples["input_ids"] = encoded_texts.input_ids
    examples["attention_mask"] = encoded_texts.attention_mask
    return examples


def _compute_metrics(predictions: EvalPrediction) -> dict[str, float]:
    logits = predictions.predictions
    labels = predictions.label_ids
    preds = logits.argmax(axis=-1)
    acc = metrics.accuracy_score(labels, preds)
    auc = metrics.roc_auc_score(labels, preds)
    return {"accuracy": acc, "auroc": auc}


def _init_params(model):
    for child in model.children():
        if isinstance(child, PreTrainedModel):  # skip PreTrainedModel params
            continue
        for p in child.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    kaiming_uniform_(p)
                else:
                    stdv = 1. / (p.shape[0] ** 0.5)
                    nn.init.uniform_(p, a=-stdv, b=stdv)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dt", "--train_dataset_path", type=str, required=True, )
    parser.add_argument("-dv", "--val_dataset_path", type=str, required=True)
    parser.add_argument("-p", "--pretrained_path", type=str, required=True, help="e.g. roberta-base")
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("-l", "--logging_dir", type=str, required=True)
    parser.add_argument("--eval_strategy", default="epoch", type=str, help="'steps' or 'epoch'")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="multiple of accumulation steps")
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--num_epochs", default=1, type=int)
    parser.add_argument("--optimizer_name", default="adamw", type=str)
    parser.add_argument("--scheduler_type", default="cosine", type=str)
    parser.add_argument("--warmup_ratio", default=0.03, type=float, help="try 0.03 for bert")
    parser.add_argument("--logging_steps", default=50, type=int)
    parser.add_argument("--eval_steps", default=50, type=int, help="only valid when strategy is 'steps'")
    parser.add_argument("--save_strategy", default="best", type=str, help="recommend 'best' or 'no'")
    parser.add_argument("--attention_dropout", default=0, type=float)
    parser.add_argument("--hidden_dropout", default=0, type=float)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    roberta_config = RobertaConfig.from_pretrained(args.pretrained_path)
    roberta_config.attention_probs_dropout_prob = args.attention_dropout
    roberta_config.hidden_dropout_prob = args.hidden_dropout
    tokenizer = RobertaTokenizerFast.from_pretrained(args.pretrained_path)
    model = RobertaForSequenceClassification.from_pretrained(args.pretrained_path, config=roberta_config).to(0)
    _init_params(model)

    dataset_files = {
        "train": args.train_dataset_path,
        "validation": args.val_dataset_path
    }
    dataset = load_dataset("json", data_files=dataset_files)
    dataset = dataset.map(partial(preprocess, tokenizer=tokenizer), batched=True, batch_size=None)
    train_dataset, val_dataset = dataset["train"], dataset["validation"]

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy=args.eval_strategy,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=1e-2,
        num_train_epochs=args.num_epochs,
        lr_scheduler_type=args.scheduler_type,
        warmup_ratio=args.warmup_ratio,
        log_level="info",
        logging_dir=args.logging_dir,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_total_limit=1,
        save_only_model=True,
        metric_for_best_model="accuracy",
        report_to="swanlab"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=_compute_metrics,
    )

    trainer.train()
