import json
import asyncio
from configparser import ConfigParser
import uvicorn
from fastapi import FastAPI, WebSocket

from barrage import DYBarrageBuilder
from logger import logger
from utils import TimeUtil
from bot import CocoBotHook, CocoBot

app = FastAPI()

ready = False

# 读取配置文件
config = ConfigParser()
config.read('config.ini', encoding="utf-8")
# 获取端口信息
port = int(config.get('app', 'port'))
# 获取直播平台
live = config.get('app', 'live')
# 消息队列
message_queue = asyncio.Queue()
# 免费到期时间
date = TimeUtil.toDate(config.get("app", "date"))

bot_websocket: WebSocket | None = None  # WebSocket


# 机器人初始化
class Hook(CocoBotHook):
    async def onConsume(self, text: str, emotion: dict, wav_byte: bytes):
        print("播放语音")
        expression = max(emotion, key=emotion.get)
        try:
            await bot_websocket.send_text("#{0}".format(text))
            await bot_websocket.send_text("${0}".format(expression))
            await bot_websocket.send_bytes(wav_byte)
        except AssertionError:
            pass

    async def onStart(self, action):
        try:
            await bot_websocket.send_text("1000")
        except AssertionError:
            pass

    async def noSing(self, name: str):
        try:
            await bot_websocket.send_text(f"1002|抱歉！没有找到音频{name}")
        except AssertionError:
            pass

    async def onLoadSing(self, name: str):
        print("开始加载音频")
        try:
            await bot_websocket.send_text(f"1003|正在加载{name}，预估时间：30s")
        except AssertionError:
            pass

    async def onSing(self, name: str, vocals_byte: bytes, wav_byte: bytes):
        print("开始传输音频")
        try:
            await bot_websocket.send_text(f"1004|{name}")
            await bot_websocket.send_bytes(vocals_byte)
            await bot_websocket.send_bytes(wav_byte)
        except AssertionError:
            pass

    async def onConsumeDown(self):
        try:
            await bot_websocket.send_text("1001")
        except AssertionError:
            pass


chatbot: CocoBot

NAME = "纳西妲"

url = "https://live.douyin.com/639709145929"


@app.on_event("startup")
async def startup():
    global chatbot, message_queue

    chatbot = CocoBot(config)
    chatbot.bind(Hook())

    async def callback(message):
        message = json.loads(message)
        await message_queue.put(message)

    barrage = DYBarrageBuilder().port(8080).page(url).on(callback).build()
    barrage.daemon = True
    barrage.start()


@app.websocket("/bot")
async def websocket_endpoint(websocket: WebSocket):
    global ready
    await websocket.accept()
    ready = True
    logger.info("cocoBot 机器人连接成功")

    global bot_websocket
    bot_websocket = websocket

    users = {}
    gift_json = config.get("app", "gift")
    gifts = json.loads(gift_json)
    while True:
        message = await message_queue.get()
        logger.info(message)
        userid = message["user_id"]
        if message["gift"]:
            users[userid] += gifts.get("gift_name", 0) * message["gift_number"]
            await chatbot.chat(message["describe"])
        elif message["join"]:
            await chatbot.chat(message["nickname"] + "加入直播间")
        elif message["message"] and str(message["content"]).startswith(NAME):
            if users.get(userid) and users[userid] > 0 or TimeUtil.is_expired(date):
                if await chatbot.chat(str(message["content"])[len(NAME):]):
                    users[userid] -= 1


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
