from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 使用 DeepSeek 官方模型
model_id = "deepseek-ai/deepseek-llm-7b-chat"

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)

# 设置 tokenizer 参数
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 构造 prompt（输入内容）
prompt = "请告诉我明天是几月几号，你可以通过"

# 编码输入
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 生成推理结果
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
        temperature=0.1,
    )

# 解码输出
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n🕒 模型推理结果：\n")
print(response)
# # import datetime

# # # 定义当前日期和时间
# # now = datetime.datetime.now()
# # print(now.weekday())
# # # 定义要转换的日期和时间
# # next_wednesday_11am = now + datetime.timedelta(days=(6-now.weekday()) + 3, hours=11)

# # # 将日期和时间转换为UTC格式
# # utc_format = next_wednesday_11am.strftime('%Y-%m-%dT%H:%M:%SZ')

# # print(utc_format)
