# tools/__init__.py
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from .calendar import create_meeting

class MeetingArgs(BaseModel):
    summary: str               = Field(..., description="会议主题")
    start_utc: str             = Field(..., description="ISO 格式 UTC 开始时间")
    duration_min: int          = Field(..., description="会议时长(分钟)")
    attendees_emails: list[str] = Field(..., description="参会人邮箱列表")

create_meeting_tool = StructuredTool.from_function(
    func=create_meeting,
    name="create_meeting",
    args_schema=MeetingArgs,
    description="根据参数创建会议并返回结果"
)
