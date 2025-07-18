# prompt_templates.py
from datetime import datetime

# —— 可选：系统提示，放在 Agent 初始化处 ——
SYSTEM_TEMPLATE = f"""
你是一个智能会议助手。你需要从用户的描述中提取会议信息，
并 **严格输出** 单段 JSON，用于调用工具。
JSON 格式必须如下（不要多余文字）：

{{
  "name": "create_meeting",
  "arguments": {{
    "title":                 string,           // 会议主题
    "start_time":            string,           // ISO 8601，含 +09:00
    "duration_minutes":      int,              // 时长(分钟)
    "attendees":             string[]          // 邮箱数组
  }}
}}
如果信息不足，请输出：
{{"name":"none","arguments":{{"reason":"<反问内容>"}}}}
"""

def build_prompt(now_dt: datetime, user_request: str) -> str:
    """
    返回模型最终 prompt，保持与训练阶段一致：
    <|user|>\n### Instruction:\n{instruction}\n\n<|assistant|>\n
    """
    now_str = now_dt.strftime("%Y-%m-%d（%A）%H:%M")
    instruction = (
        "你是一个智能会议助手，根据我提供的信息提取出title, start time, duration minutes, attendees。"f"当前时间是 {now_str}，{user_request}"
    )

    prompt = (
        "<|user|>\n### Instruction:\n"
        f"{instruction}\n\n"
        "<|assistant|>\n"
    )
    return prompt
