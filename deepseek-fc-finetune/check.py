from datasets import load_dataset
from transformers import AutoTokenizer
import json, textwrap

# ============ 定义 format_sample ============
def format_sample(example_or_batch):
    is_batched = isinstance(example_or_batch["instruction"], list)

    def _one(instr, steps):
        step_txt = json.dumps(steps, ensure_ascii=False, indent=0)
        return f"### Instruction:\n{instr}\n\n### Output:\n{step_txt}\n"

    if not is_batched:
        return [_one(example_or_batch["instruction"], example_or_batch["output"])]
    else:
        return [_one(i, o) for i, o in zip(example_or_batch["instruction"], example_or_batch["output"])]

# ============ 加载数据集 ============
ds = load_dataset("json", data_files="data/train.jsonl")["train"]
print("✅ Dataset loaded:", len(ds), "samples")

# ============ 加载 tokenizer ============
tok = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-chat", trust_remote_code=True)
tok.pad_token = tok.eos_token

# ============ 全量检查函数 ============
def full_sanity_check(dataset):
    bad_control_count = 0
    unk_token_count = 0

    for idx, ex in enumerate(dataset):
        prompt = format_sample(ex)[0]
        bad_chars = [c for c in prompt if ord(c) < 32 and c not in ['\n', '\t']]
        if bad_chars:
            print(f"[{idx}] ⚠️ 控制字符: {bad_chars}")
            bad_control_count += 1

        tokens = tok(prompt)["input_ids"]
        unk = tokens.count(tok.unk_token_id)
        if unk > 0:
            print(f"[{idx}] ⚠️ UNK tokens: {unk} → {textwrap.shorten(prompt, 100)}")
            unk_token_count += 1

    print("\n📊 检查完成")
    print("————————————————————")
    print("控制字符异常样本数量:", bad_control_count)
    print("包含 UNK token 的样本数量:", unk_token_count)
    print("总样本数量:", len(dataset))
    print("————————————————————")

# ============ 执行检查 ============
full_sanity_check(ds)
