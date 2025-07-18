# main.py
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# ────────────────────────── 1. 数据（纯 JSON → Dataset） ─────────────────────────
# data_path = "deepseek-fc-finetune/data/train.jsonl"        # ← 你的原始文件
data_path = "data/train.jsonl"
raw_ds = load_dataset("json", data_files=data_path)["train"]

# ────────────────────────── 2. Tokenizer ────────────────────────────────────────
base_model_id = "deepseek-ai/deepseek-llm-7b-chat"
tok = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
tok.pad_token = tok.eos_token
tok.padding_side = "right"

# ────────────────────────── 3. 格式化函数 (必填) ────────────────────────────────
def formatting_func(example):
    """
    将一条原始样本拼成纯文本 prompt：
    ### Instruction:
    ...
    
    ### Output:
    ...
    """
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Output:\n"
    output = json.dumps(example["output"], ensure_ascii=False)
    formatted = f"<|user|>\n{prompt}\n<|assistant|>\n{output}"
    return [formatted]  # 注意要返回 list[str]

# ────────────────────────── 4. 基座模型 + LoRA ─────────────────────────────────
base = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    torch_dtype="auto",          # 自动 FP16 / BF16 取决于 GPU
    trust_remote_code=True,
)

lora_cfg = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(base, lora_cfg)
model.enable_input_require_grads()
model.gradient_checkpointing_enable()
model.config.use_cache = False

# ────────────────────────── 5. 训练参数 ─────────────────────────────────────────
train_args = TrainingArguments(
    output_dir="output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=5e-5,
    fp16=False, bf16=False,              # 先用纯 FP32；跑通再酌情改 FP16
    logging_steps=5,
    remove_unused_columns=False,         # 必须，否则 labels 会被删
    max_grad_norm=0.3,
)

# ────────────────────────── 6. SFTTrainer (老版本用法) ─────────────────────────
trainer = SFTTrainer(
    model=model,
    args=train_args,
    train_dataset=raw_ds,                # 注意：传 **未 token 化** 的 Dataset
    tokenizer=tok,
    formatting_func=formatting_func,     # ★ 关键：告诉 SFTTrainer 如何拼文本
    max_seq_length=512,                # 序列截断长度
    dataset_batch_size=1,
)

# ────────────────────────── 7. 训练 & 保存 ──────────────────────────────────────
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in model.parameters())
print(f"🔍 可训练参数数量: {n_trainable:,} / {n_total:,} ({n_trainable / n_total:.4%})")
trainer.train()
trainer.save_model("output/deepseek-lora")
