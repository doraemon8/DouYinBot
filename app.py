import json
import asyncio
from configparser import ConfigParser
from json import JSONDecodeError

import uvicorn
from fastapi import FastAPI, WebSocket

from logger import logger
from utils import TimeUtil
from bot import CocoBotHook, CocoBot, Status

app = FastAPI()

ready = False

# 读取配置文件
config = ConfigParser()
config.read('config.ini', encoding="utf-8")
# 获取端口信息
port = int(config.get('app', 'port'))
# 获取直播平台
live = config.get('app', 'live')
# 获取唤醒关键词
word = config.get("app", 'wake')
# 唱歌唤醒关键词
word2 = ["来一首", "唱一下", "我想听"]
# 消息队列
message_queue = asyncio.Queue()
# 免费到期时间
date = TimeUtil.toDate(config.get("app", "date"))
# 直播礼物
# gifts = config.get("app", "gift")
# gifts = json.loads(gifts)
# 机器人接口
bot_websocket: WebSocket | None = None  # WebSocket
# 直播接口
live_websocket: WebSocket | None = None
# 机器人实例化
chatbot: CocoBot


# 机器人钩子
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
            print("音频传输完成")
        except AssertionError:
            pass


@app.on_event("startup")
async def startup():
    global chatbot
    chatbot = CocoBot(config).bind(Hook())


@app.get("/ok")
async def play_ok():
    chatbot.switch(Status.ONE)
    print("客户端音乐播放完成")


@app.websocket("/live")
async def websocket_endpoint(websocket: WebSocket):
    global live_websocket

    await websocket.accept()

    live_websocket = websocket
    print("直播接入成功")

    while True:
        message = await websocket.receive_text()
        if ready:
            try:
                message = json.loads(message)
                await message_queue.put(message)
            except JSONDecodeError:
                logger.error("{}解码异常", message)


@app.websocket("/bot")
async def websocket_endpoint(websocket: WebSocket):
    global bot_websocket, ready

    await websocket.accept()

    bot_websocket = websocket
    ready = True
    print("cocoBot 机器人连接成功")

    while True:
        message = await message_queue.get()
        logger.info(message)
        action = message["action"]
        if action == "gift":
            chatbot.chat(message["describe"])
        elif action == "join":
            chatbot.chat(message["nickname"] + "加入直播间")
        elif action == "message" and str(message["content"]).startswith(word):
            message = str(message["content"])[len(word):]
            if message[0:3] in word2:
                chatbot.sing(message[3:])
            else:
                chatbot.chat(message)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=port)
