from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training
import torch, os

# 1. KorQuAD로 파인튜닝+병합된 모델을 베이스로 불러오기
base_model = "./merged-1B-instruct-ko"   # 앞에서 merge한 모델 경로
output_dir = "./llama-1B-will-finetuned"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. 우리 데이터셋 로드
dataset = load_dataset(
    "json",
    data_files={
        "train": "/app/dataset/train.jsonl",
        "validation": "/app/dataset/validation.jsonl",
    }
)

# 3. formatting_func (데이터 구조 맞게 수정해야 함)
def formatting_func(examples):
    texts = []
    for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
        messages = [
            {"role": "system", "content": "아래 지시에 따라 답변하세요."},
            {"role": "user", "content": f"{inst}\n{inp}" if inp else inst},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        response = out.strip() + tokenizer.eos_token
        texts.append(prompt + response)
    return {"text": texts}

train_proc = dataset["train"].map(formatting_func, batched=True, remove_columns=dataset["train"].column_names)
eval_proc = dataset["validation"].map(formatting_func, batched=True, remove_columns=dataset["validation"].column_names)

# 4. LoRA 설정 (추가 학습)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj"]
)

# 5. 학습 파라미터
training_arguments = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    bf16=True,
    max_steps=200,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=20,
    save_strategy="epoch",
    dataset_text_field="text",
    packing=True,
)

# 6. Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_proc,
    eval_dataset=eval_proc,
    peft_config=peft_config,
    args=training_arguments,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
