import dateparser
from datetime import datetime, timezone

def parse_to_utc(text: str, tz="Asia/Shanghai"):
    # 以当前时间为参考解析
    dt = dateparser.parse(
        text,
        languages=["zh"],
        settings={
            "TIMEZONE": tz,
            "RETURN_AS_TIMEZONE_AWARE": True,
        },
    )
    if dt is None:
        raise ValueError(f"无法识别时间表达式: {text}")
    return dt.astimezone(timezone.utc)      # 转 UTC

print(parse_to_utc("Friday"))
# 例如输出: 2025-06-18 03:00:00+00:00  (假设今天是 2025-06-09)
