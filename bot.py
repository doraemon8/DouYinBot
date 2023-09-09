import asyncio
import os
import pathlib
import pickle
import threading
import time
from abc import ABC, abstractmethod
from asyncio import Queue
from configparser import ConfigParser
from enum import Enum

import jieba

from chatgpt import Chatbot
from logger import logger
from music import get_music_mp3_url, generate_unique_code
from vc import VCBot
from vits import TTSBot


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


class Status(Enum):
    ONE = "空闲"
    TWO = "文字未生成完成,语音未生成完成"
    THREE = "文字生成完成,语音未生成完成"
    FOUR = "歌声未生成完成"
    FIVE = "歌声生成完成"


class Emotion(object):

    def __init__(self):
        self.Haos = self.read_dict('好.pkl')
        self.Les = self.read_dict('乐.pkl')
        self.Ais = self.read_dict('哀.pkl')
        self.Nus = self.read_dict('怒.pkl')
        self.Jus = self.read_dict('惧.pkl')
        self.Wus = self.read_dict('恶.pkl')
        self.Jings = self.read_dict('惊.pkl')

    def read_dict(self, file):
        pathchain = ['dictionary', file]
        mood_dict_filepath = pathlib.Path(__file__).parent.joinpath(*pathchain)
        dict_f = open(mood_dict_filepath, 'rb')
        words = pickle.load(dict_f)
        return words

    def infer(self, text):
        hao, le, ai, nu, ju, wu, jing = 0, 0, 0, 0, 0, 0, 0
        words = jieba.lcut(text)
        for w in words:
            if w in self.Haos:
                hao += 1
            elif w in self.Les:
                le += 1
            elif w in self.Ais:
                ai += 1
            elif w in self.Nus:
                nu += 1
            elif w in self.Jus:
                ju += 1
            elif w in self.Wus:
                wu += 1
            elif w in self.Jings:
                jing += 1
            else:
                pass
        result = {'好': hao, '乐': le, '哀': ai, '怒': nu, '惧': ju, '恶': wu, '惊': jing}
        return result


class Action:
    SING = 1
    CHAT = 0


class CocoBotHook(ABC):
    async def onStart(self, action):
        pass

    async def onBusy(self):
        pass

    @abstractmethod
    async def onConsume(self, text: str, emotion: dict, wav_byte: bytes):
        pass

    async def onText(self, sentence: str):
        pass

    async def onSpeech(self, sentence: str):
        pass

    async def onConsumeDown(self):
        pass

    async def onTextDown(self, text: str):
        pass

    async def onSpeechDown(self):
        pass

    async def onEnd(self):
        pass

    async def noSing(self, name: str):
        pass

    async def onLoadSing(self, name: str):
        pass

    @abstractmethod
    async def onSing(self, name: str, vocals_byte: bytes, wav_byte: bytes):
        pass

    async def onLoadSingDown(self, name: str):
        pass

    async def onLoadSingBack(self, name: str):
        pass


# 创建异步线程函数
def create_async_thread(func, *args):
    if asyncio.iscoroutinefunction(func):
        def async_run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(func(args))
            loop.close()

        return threading.Thread(target=async_run, args=args)
    return threading.Thread(target=func, args=args)





class CocoBot:

    def __init__(self, config: ConfigParser):
        self.sign = "####"
        self.status = Status.ONE
        self.tts = TTSBot(config.get("tts", "path"), config.get("tts", "config"), config.get("tts", "name"))
        if config.get("chatgpt", "access_token"):
            self.gpt = Chatbot(
                config={"access_token": config.get("chatgpt", "access_token")},
                conversation_id=config.get("chatgpt", "conversation_id"))
        else:
            self.gpt = Chatbot(
                config={"email": config.get("chatgpt", "email"), "password": config.get("chatgpt", "password")},
                conversation_id=config.get("chatgpt", "conversation_id"))
        self.emotion = Emotion()
        self.text_queue = Queue()
        self.audio_queue = Queue()
        self.hook = None
        self.lock = threading.Lock()

    def __init(self):
        if not os.path.exists(self.temp_dir):
            os.mkdir(self.temp_dir)
        if not os.path.exists(self.temp2_dir):
            os.mkdir(self.temp2_dir)
        if not os.path.exists(self.music_dir):
            os.mkdir(self.music_dir)
        if not os.path.exists(self.music_dir2):
            os.mkdir(self.music_dir2)
        if not self.vc.isGPU:
            thread = threading.Thread(target=self._singing_voice_synthesis_back)
            thread.start()

    def bind(self, hook: CocoBotHook):
        self.hook = hook
        return self

    async def chat(self, prompt: str):
        if self.status is not Status.ONE:
            await self.hook.onBusy()
            return False
        else:
            text_generation_task = create_async_thread(self._text_generation, prompt)
            speech_generation_task = create_async_thread(self._speech_generation)
            audio_consume_task = create_async_thread(self._audio_consume)
            text_generation_task.start()
            speech_generation_task.start()
            audio_consume_task.start()
            self.switch_status(Status.TWO)
            await self.hook.onStart(Action.CHAT)
            return True

    async def _text_generation(self, prompt):
        print(f"start to text {prompt}")
        prev_text = ""
        symbol = ["。", "？", "!", "：", "；", "！"]
        sentence = ""
        for data in self.gpt.ask(
                prompt,
        ):
            message = data["message"][len(prev_text):]
            prev_text = data["message"]
            sentence += message
            if message in symbol:
                print(f'put text {sentence}')
                await self.text_queue.put(sentence)
                sentence = ""
                await self.hook.onText(sentence)
        await self.text_queue.put(self.sign)
        self.switch_status(Status.THREE)
        await self.hook.onTextDown(prev_text)

    def switch_status(self, statu: Status):
        with self.lock:
            self.status = statu

    async def _speech_generation(self):
        while True:
            text = await self.text_queue.get()
            # 收到结束信号，返回
            if text == self.sign:
                break
            logger.info("出列消息>>{}".format(text))
            start = time.time()
            wav_byte = self.tts.infer(text)
            await self.hook.onSpeech(text)
            logger.info("推理用时>>{}".format(time.time() - start))
            data = {"text": text, "wav_byte": wav_byte}
            await self.audio_queue.put(data)
            await self.audio_queue.put(self.sign)
        await self.hook.onSpeechDown()

    async def _audio_consume(self):
        while True:
            data = await self.audio_queue.get()
            if not isinstance(data, dict):
                # 结束信号
                break
            emotion = self.emotion.infer(data["text"])
            self.hook.onConsume(data["text"], emotion, data["wav_byte"])
        self.switch_status(Status.ONE)
        await self.hook.onConsumeDown()


