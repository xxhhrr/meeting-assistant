# tools/calendar.py
from typing import List, Dict

def create_meeting(summary: str,
                   start_utc: str,
                   duration_min: int,
                   attendees_emails: List[str]) -> Dict:
    """
    用真实日历 API 时替换这里的逻辑。
    现在先返回模拟数据，方便验证 Agent 是否能正确调用。
    """
    return {
        "result":   "已创建会议",
        "summary":  summary,
        "start":    start_utc,
        "duration": duration_min,
        "attendees": attendees_emails
    }
