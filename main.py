from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import json

# 加载模型
base_model_id = "deepseek-ai/deepseek-llm-7b-chat"
tok = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
tok.pad_token = tok.eos_token

base = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto", torch_dtype="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(base, "deepseek-fc-finetune/output/deepseek-lora")

# 构造推理 prompt

instruction = "你是一个智能会议助手，根据我提供的信息提取出title，start time，duration minutes，attendees。当前时间是 2025-06-16（(Monday)）10:40，请安排下周三晚上八点和马俊飞开一个云南旅游计划商议，持续60分钟。"
prompt = f"<|user|>\n### Instruction:\n{instruction}\n\n<|assistant|>\n"

# 推理
input_ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
with torch.no_grad():
    outputs = model.generate(input_ids, max_new_tokens=150)

# 解码结果
response = tok.decode(outputs[0], skip_special_tokens=True)
print("🧾 模型输出：")
print(response)
# main.py
# import datetime, sys
# from agent.agent_builder import get_meeting_agent
# from prompt_templates import build_prompt

# agent = get_meeting_agent()

# if __name__ == "__main__":
#     if len(sys.argv) > 1:
#         user_req = " ".join(sys.argv[1:])
#     else:
#         user_req = input("❓ 请输入会议请求：")

#     now_str = datetime.datetime.now()hash -r

#     final_prompt = build_prompt(now_str, user_req)

#     result = agent.invoke(final_prompt)
#     print("🔹 Agent 返回：", result)
# import json, datetime
# from langchain.output_parsers import JsonOutputParser
# from langchain.schema import StrOutputParser
# from langchain.schema.runnable import RunnableLambda, RunnablePassthrough, RunnableSequence
# from llm.local_llm import load_local_chat
# from prompt_templates import SYSTEM_TEMPLATE, build_prompt
# from tools import create_meeting_tool

# # ① 固定 system + 用户 prompt
# def build_messages(user_request:str):
#     now_str = datetime.datetime.now().strftime("%Y-%m-%d（%A）%H:%M")
#     prompt  = build_prompt(now_str, user_request)
#     return [ {"role":"system","content": SYSTEM_TEMPLATE},
#              {"role":"user",  "content": prompt} ]

# # ② LLM
# llm = load_local_chat()

# # ③ JsonOutputParser（确保严格 JSON）
# parser = JsonOutputParser()

# # ④ Tool executor  (wrap StructuredTool.invoke)
# def call_tool(data: dict):
#     if data.get("name") == "create_meeting":
#         return create_meeting_tool.invoke(data["arguments"])
#     else:
#         return data["arguments"]["reason"]

# # ⑤ 串成可执行链
# chain: RunnableSequence = (
#     RunnablePassthrough()                   # 传入 user_request
#     | RunnableLambda(lambda req: build_messages(req))   # 构造 messages
#     | llm                                    # 生成
#     | StrOutputParser()                      # 取纯字符串
#     | parser                                 # 解析 JSON
#     | RunnableLambda(call_tool)              # 调用工具
# )

# if __name__ == "__main__":
#     user_req = input("❓ 请输入会议请求：")
#     result = chain.invoke(user_req)
#     print("🔧 最终结果:", result)
