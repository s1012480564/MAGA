import argparse

from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    default_data_collator,
    BitsAndBytesConfig,
    is_torch_tpu_available,
    set_seed,
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from model import load_model, load_tokenizer
import torch
from datasets import load_dataset

def filter_by_english_and_llama3_mistral(sample):
    return len(sample["conversation"][0]["content"]) < 480

def filter_by_english_and_version_3_5(sample):
    return sample["language"] == 'English' \
            and sample["model"] == "gpt-3.5-turbo" and len(sample["conversation"][0]["content"]) < 480

def filter_by_english_and_version(sample):
    return sample["language"] == 'English' and  sample["timestamp"].month <= 6 \
            and sample["model"] == "gpt-4" and len(sample["conversation"][0]["content"]) < 512

def filter_claude(sample):
    return len(sample["prompt"]) < 2048

def filter_claude3(sample):
    return len(sample["conversation"][0]["content"]) < 1024

filter_dict = {
    'ChatGPT':filter_by_english_and_version_3_5,
    'GPT-4':filter_by_english_and_version,
    'Claude-3':filter_claude3
}


def build_sft_dataset(name, num_sample, filter_fn=None):
    if name == "wildchat":
        dataset = load_dataset("allenai/WildChat", split="train")
        if filter_fn is not None:
            dataset = dataset.filter(filter_fn)
    elif name == "claude":
        dataset = load_dataset("Sao10K/Claude-3-Opus-Instruct-5K", 'Instruct Data v1 - Merged', split="train")
        if filter_fn is not None:
            dataset = dataset.filter(filter_fn)
    elif name == "claude3":
        dataset = load_dataset("Shengkun/Claude3-texts", split="train")
        if filter_fn is not None:
            dataset = dataset.filter(filter_fn)
    elif name == "llama3":
        dataset = load_dataset("Shengkun/llama3_texts", split="train")
        if filter_fn is not None:
            dataset = dataset.filter(filter_fn)
    elif name == "mistral":
        dataset = load_dataset("Shengkun/Mistral_texts", split="train")
        if filter_fn is not None:
            dataset = dataset.filter(filter_fn)
    return dataset.train_test_split(train_size=num_sample)["train"]


def train(args):

    sft_training_data = build_sft_dataset(name=args.train_dataset_name, num_sample=args.num_samples, filter_fn=filter_dict[args.target_model_name])

    tokenizer = load_tokenizer(args.scoring_model_name, args.train_dataset_name, args.cache_dir) 
    #generate prompt
    cutoff_len = 1024
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding="max_length",
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result
    
    def generate_instruct_token(prompt, prompt_response_comb):
        prompts_ids = tokenizer(
            text=prompt,
            return_tensors=None,
        )

        combined_target_ids = tokenizer(
            text=prompt_response_comb,
            return_tensors=None,
            padding="max_length", max_length=cutoff_len, truncation=True
        )                    

        prompt_len = len(prompts_ids["input_ids"])
        labels = [-100] * prompt_len + combined_target_ids["input_ids"][prompt_len:]
        combined_target_ids["labels"] = labels

        return combined_target_ids

    def generate_gpt_input_and_tokenize(sample):
        conversation = sample["conversation"]
        prompt = conversation[0]['content']
        prompt_response = f"{conversation[0]['content']}{conversation[1]['content']}"
        return generate_instruct_token(prompt=prompt, prompt_response_comb=prompt_response)
        
    def generate_claude_input_and_tokenize(sample):
        prompt = sample["prompt"]
        prompt_response = f"{sample['prompt']}{sample['response']}"
        return generate_instruct_token(prompt=prompt, prompt_response_comb=prompt_response)

    def generate_claude3_input_and_tokenize(sample):
        prompt = sample["claude3_prompt"]
        prompt_response = f"{sample['claude3_prompt']}{sample['claude3_output']}"
        return generate_instruct_token(prompt=prompt, prompt_response_comb=prompt_response)

    def generate_llama3_input_and_tokenize(sample):
        prompt = sample["llama3_8b_prompt"]
        prompt_response = f"{sample['llama3_8b_prompt']}{sample['llama3_8b_output']}"
        return generate_instruct_token(prompt=prompt, prompt_response_comb=prompt_response)
    
    def generate_mistral_input_and_tokenize(sample):
        prompt = sample["mistral-7b_prompt"]
        prompt_response = f"{sample['mistral-7b_prompt']}{sample['mistral-7b_output']}"
        return generate_instruct_token(prompt=prompt, prompt_response_comb=prompt_response)

    tokenized_dataset = sft_training_data.map(generate_gpt_input_and_tokenize)

    tokenized_dataset = tokenized_dataset.remove_columns(sft_training_data.column_names)


    model = load_model(args.scoring_model_name, device="cpu", cache_dir=args.cache_dir)
    modules = {
        "llama3-8b":["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        "llama2-7b":["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        "Gpt-neo":["q_proj", "v_proj", "k_proj", "out_proj", "c_fc", " c_proj"]
    }

    # load LoRA model
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=modules[args.scoring_model_name],
        fan_in_fan_out=False,
        lora_dropout=0.05,
        inference_mode=False,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    #Train and Evaluate
    args = TrainingArguments(
        output_dir=args.output_model_dir,
        remove_unused_columns=False,
        save_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        bf16=True,
        num_train_epochs=1,
        logging_steps=1,
        do_eval=False,
        )
    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, return_tensors="pt", padding=False),
        callbacks=[]
        )
    torch.cuda.empty_cache()
    trainer.train()

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset_name', type=str, default="wildchat")
    parser.add_argument('--target_model_name', type=str, default="GPT-4")
    parser.add_argument('--num_samples', type=int, default=5000)
    parser.add_argument('--scoring_model_name', type=str, default="llama3-8b")
    parser.add_argument('--output_model_dir', type=str, default="./ckpt")
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()
    
    train(args)