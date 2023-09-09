from datetime import datetime


class TimeUtil:
    @staticmethod
    def toDate(data_str: str):
        datetime.strptime(data_str, "%Y-%m-%d")

    @staticmethod
    def is_expired(date):
        # 获取当前日期
        current_date = datetime.now().date()
        # 比较目标日期和当前日期
        if date < current_date:
            return True
        else:
            return False
