# 🧠 Smart Meeting Assistant · 智能会议助手

一个基于中文大语言模型的轻量级会议助手，支持从自然语言中**自动提取会议标题、起始时间、持续时间和参会人员**等结构化信息，辅助实现会议创建、日程管理等自动化办公功能。

> ✨ 基于 [DeepSeek-LLM-7B-Chat](https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat)，通过 LoRA 微调和结构化指令优化。

---

## 📌 项目特性

- 🗣️ 中文自然语言指令理解（如“下下周一下午六点”）
- 📤 输出结构化 JSON 格式，便于系统对接
- 🧠 理解并解析模糊表达，如“明天下午”、“后天晚上”
- 📅 易于接入日历/会议系统（Google Calendar, Outlook 等）
- ⚙️ 支持二阶段指令微调 + 格式对齐优化

---

## 🖥️ 示例
<img width="732" height="313" alt="image" src="https://github.com/user-attachments/assets/42956180-74a6-4bca-ba49-49012b1da975" />

### 输入示例

