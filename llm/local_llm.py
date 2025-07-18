# llm/local_llm.py
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from langchain.schema import (
    AIMessage, BaseMessage, ChatGeneration, ChatResult
)
from langchain.chat_models.base import BaseChatModel


_BASE_ID  = "deepseek-ai/deepseek-llm-7b-chat"
_LORA_DIR = "deepseek-fc-finetune/output/deepseek-lora-stage2"

class ChatLocal(BaseChatModel):
    """本地 LoRA Chat，供 LangChain Agent 使用（方案 A）"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)            # 先让 Pydantic 初始化
        # tok   = AutoTokenizer.from_pretrained(_LORA_DIR, trust_remote_code=True)
        # base  = AutoModelForCausalLM.from_pretrained(
        #     _BASE_ID, device_map="auto", torch_dtype="auto", trust_remote_code=True
        # )
        # model = PeftModel.from_pretrained(base, _LORA_DIR)
        tok = AutoTokenizer.from_pretrained(_BASE_ID, trust_remote_code=True)
        tok.pad_token = tok.eos_token

        base = AutoModelForCausalLM.from_pretrained(_BASE_ID, device_map="auto", torch_dtype="auto", trust_remote_code=True)
        model = PeftModel.from_pretrained(base, _LORA_DIR)

        # 用 object.__setattr__ 绕开 Pydantic 的字段检查
        object.__setattr__(self, "tok", tok)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "eos_id", tok.eos_token_id)

    # ---- 必须实现：核心生成逻辑 ----
    def _generate(self, messages: List[BaseMessage], **_) -> AIMessage:
        prompt = "".join(msg.content for msg in messages)
        ids    = self.tok(prompt, return_tensors="pt").to(self.model.device)
        # out_ids = self.model.generate(
        #     **ids,
        #     max_new_tokens=128,
        #     do_sample=False,
        #     eos_token_id=self.eos_id,
        # )
        with torch.no_grad():
            out_ids = self.model.generate(**ids, max_new_tokens=150)
        resp = self.tok.decode(out_ids[0], skip_special_tokens=True).strip()
        ai_msg = AIMessage(content=resp)

        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    @property
    def _llm_type(self) -> str:
        return "local_lora"

def load_local_chat():
    return ChatLocal()
