import json

input_path = "data/meeting_synthetic_train.jsonl"
output_path = "data/meeting_synthetic_train_fixed.jsonl"

with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        obj = json.loads(line)
        if not isinstance(obj["output"], str):
            obj["output"] = json.dumps(obj["output"], ensure_ascii=False)
        json.dump(obj, outfile, ensure_ascii=False)
        outfile.write("\n")

print("✅ 修复完成：", output_path)
