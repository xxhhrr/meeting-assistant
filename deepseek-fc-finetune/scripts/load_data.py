import json
from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_preprocess_dataset(
    model_name="deepseek-ai/deepseek-llm-7b-chat",
    data_path="data/train.jsonl",
    max_length=512
):
    # 初始化 tokenizer
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    ds = load_dataset("json", data_files=data_path)["train"]
    bad_indices = []

    def preprocess(example, idx=None):
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Output:\n"
        output = json.dumps(example["output"], ensure_ascii=False)
        full_text = prompt + output

        tokenized = tok(
            full_text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

        prompt_len = len(tok(prompt, add_special_tokens=False)["input_ids"])
        if prompt_len >= max_length:
            # 仅保留最后 max_length-1 token 作为 prompt
            prompt_ids = prompt_ids[-(max_length-1):]
            prompt_len = len(prompt_ids)
        labels = [-100] * prompt_len + tokenized["input_ids"][prompt_len:]
        labels = labels[:max_length]
        tokenized["labels"] = labels

        # # ✅ 添加 debug 信息
        # tokens = tok.convert_ids_to_tokens(tokenized["input_ids"])
        # print("\n--- DEBUG ---")
        # for i, (tok_str, label_id) in enumerate(zip(tokens, labels)):
        #     flag = "[LOSS]" if label_id != -100 else ""
        #     print(f"{i:4d} | {tok_str:<15} | {label_id:<5} {flag}")

        # ✅ Sanity check：labels 是否全为 -100？
        if all(l == -100 for l in labels):
            bad_indices.append(idx)

        return tokenized

    # 加入索引信息以供定位问题样本
    ds = ds.add_column("idx", list(range(len(ds))))
    tokenized_ds = ds.map(lambda x: preprocess(x, idx=x["idx"]), remove_columns=ds.column_names)

    # # 输出警告
    # if bad_indices:
    #     print(f"⚠️ 警告：发现 {len(bad_indices)} 条样本的 labels 全为 -100！索引如下：\n{bad_indices}")
    # else:
    #     print("✅ 所有样本均包含有效标签。")

    return tokenized_ds, tok


def formatting_func(example):
    text = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Output:\n{json.dumps(example['output'], ensure_ascii=False)}\n"
    )
    return text