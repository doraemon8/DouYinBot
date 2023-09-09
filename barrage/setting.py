import os


class Setting:
    def __init__(self, wsurl="ws://127.0.0.1:9999/live", timeinterval=1000):
        self.wsurl = wsurl
        self.timeinterval = timeinterval

    def to_dict(self):
        return {
            "wsurl": self.wsurl,
            "timeinterval": self.timeinterval
        }

root = os.path.abspath(os.path.dirname(__file__))
