# ───────────────── 0. 环境导入 ─────────────────
import json, torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments
)
from peft import PeftModel, LoraConfig, get_peft_model
from trl import SFTTrainer

# ───────────────── 1. 数据集 ───────────────────
data_path = "data/meeting_synthetic_train_fixed.jsonl"          # ← 你的“Tool-调用格式”数据
raw_ds = load_dataset("json", data_files=data_path)["train"]

# ───────────────── 2. Tok­enizer ───────────────
base_model_id = "deepseek-ai/deepseek-llm-7b-chat"
tok = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
tok.pad_token = tok.eos_token          # 必不可少
tok.padding_side = "right"
# ───────────────── 3. 格式化函数 ───────────────
def formatting_func(example):
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Output:\n"
    output = json.dumps(example["output"], ensure_ascii=False)
    return [f"<|user|>\n{prompt}\n<|assistant|>\n{output}"]

# ───────────────── 4. 模型：加载上一阶段 LoRA ─
base = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
)
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(base,lora_cfg)  # ← 已有 LoRA
model.enable_input_require_grads()
model.gradient_checkpointing_enable()
model.config.use_cache = False

# ───────────────── 5. 训练参数 ────────────────
train_args = TrainingArguments(
    output_dir="output/deepseek-lora-stage2",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=5e-5,
    fp16=False,
    bf16=False,
    logging_steps=5,
    remove_unused_columns=False,
    max_grad_norm=0.3,
)
# ───────────────── 6. SFTTrainer ──────────────
trainer = SFTTrainer(
    model=model,
    args=train_args,
    train_dataset=raw_ds,          # 未 token 化原始数据
    tokenizer=tok,                 # ← 这里需要 tok
    formatting_func=formatting_func,
    max_seq_length=512,
    dataset_batch_size=1,
)

# # ───────────────── 7. 训练 & 保存 ─────────────
# n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
# n_total = sum(p.numel() for p in model.parameters())
# print(f"🔍 可训练参数数量: {n_trainable:,} / {n_total:,} ({n_trainable / n_total:.4%})")
trainer.train()
trainer.save_model("output/deepseek-lora-stage2")  # 二阶段 LoRA