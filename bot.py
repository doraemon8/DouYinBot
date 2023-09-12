import asyncio
import os
import pathlib
import pickle
import subprocess
import threading
import time
import soundfile as sf
from abc import ABC, abstractmethod
from configparser import ConfigParser
from enum import Enum
from queue import Queue

import jieba
from pydub import AudioSegment

from chatgpt.v1 import Chatbot
from logger import logger
from music import slice_wav_file, download, get_music_mp3_url, generate_unique_code, wav_to_bytes, concat_wav_slices
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
        def async_run(*args):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(func(*args))
            loop.close()

        return threading.Thread(target=async_run, args=args)
    return threading.Thread(target=func, args=args)


def run_async_func(func, *args):
    if asyncio.iscoroutinefunction(func):
        try:
            func(*args).send(None)
        except StopIteration as e:
            pass
    else:
        func(*args)


def combined_wav(accompaniment: str, vocal: str, out_wav: str):
    print("准备读取伴奏")
    accompaniment = AudioSegment.from_wav(accompaniment)
    print("读取伴奏成功")
    print("准备读取人声")
    vocal = AudioSegment.from_wav(vocal)
    print("读取人声成功")
    # 进行音频合成
    combined = accompaniment.overlay(vocal, position=0)
    print("人声伴奏合成")
    # 保存合成后的音频
    combined.export(out_wav, format="wav")


class CocoBot:

    def __init__(self, config: ConfigParser):
        self.sign = "####"
        self.status = Status.ONE
        self.free = 0.1
        self.hook = None
        self.running = False
        self.lock = threading.Lock()
        self.audio_queue = Queue()

        # 聊天功能依赖加载
        self.tts = TTSBot(config.get("tts", "path"), config.get("tts", "config"), config.get("tts", "name"))
        if config.get("chatgpt", "access_token"):
            self.gpt = Chatbot(
                config={"access_token": config.get("chatgpt", "access_token")},
                conversation_id=config.get("chatgpt", "conversation_id"))
        else:
            self.gpt = Chatbot(
                config={"email": config.get("chatgpt", "email"), "password": config.get("chatgpt", "password")},
                conversation_id=config.get("chatgpt", "conversation_id"))
        # 这里不能用异步队列，因为每个线程的事件循环不一样
        self.text_queue = Queue()

        # 情绪
        self.emotion = Emotion()

        # 唱歌功能依赖加载
        if int(config.get("app", "enable_sing_ext")):
            self.temp_dir = config.get("vc", "temp")
            self.temp2_dir = config.get("vc", "temp2")
            self.music_dir = config.get("vc", "music")
            self.music_dir2 = config.get("vc", "music2")
            self.vc = VCBot(config.get("vc", "path"), config.get("vc", "index"), config.get("vc", "hubert"))
            self.slice_num = 30
            self._init_dir()

    def _init_dir(self):
        if not os.path.exists(self.temp_dir):
            os.mkdir(self.temp_dir)
        if not os.path.exists(self.temp2_dir):
            os.mkdir(self.temp2_dir)
        if not os.path.exists(self.music_dir):
            os.mkdir(self.music_dir)
        if not os.path.exists(self.music_dir2):
            os.mkdir(self.music_dir2)

    def bind(self, hook: CocoBotHook):
        self.hook = hook
        return self

    def _switch(self, status: Status):
        with self.lock:
            self.status = status

    def chat(self, prompt: str):
        if self.status is not Status.ONE:
            run_async_func(self.hook.onBusy)
            return False
        else:
            text_generation_task = threading.Thread(target=self._text_generation, args=(prompt,))
            speech_generation_task = threading.Thread(target=self._speech_generation)
            audio_consume_task = threading.Thread(target=self._audio_consume)
            text_generation_task.start()
            speech_generation_task.start()
            audio_consume_task.start()
            self._switch(Status.TWO)
            run_async_func(self.hook.onStart, Action.CHAT)
            return True

    def free(self):
        self._switch(Status.ONE)

    def speak(self, text):
        # 阻塞直到机器空闲执行完任务为止
        while self.status != Status.ONE:
            time.sleep(self.free)
        self._switch(Status.THREE)
        emotion = self.emotion.infer(text)
        wav_byte = self.tts.infer(text)
        run_async_func(self.hook.onConsume, text, emotion, wav_byte)

    def _text_generation(self, prompt):
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
                self.text_queue.put(sentence)
                sentence = ""
                run_async_func(self.hook.onText, sentence)
        self.text_queue.put(self.sign)
        self._switch(Status.THREE)
        run_async_func(self.hook.onTextDown, prev_text)

    def _speech_generation(self):
        while True:
            time.sleep(self.free)
            if not self.text_queue.empty():
                text = self.text_queue.get()
                if text == self.sign:
                    break
                logger.info("出列消息>>{}".format(text))
                start = time.time()
                wav_byte = self.tts.infer(text)
                run_async_func(self.hook.onSpeech, text)
                logger.info("推理用时>>{}".format(time.time() - start))
                data = {"type": "chat", "text": text, "wav_byte": wav_byte}
                self.audio_queue.put(data)
        self.audio_queue.put(self.sign)
        run_async_func(self.hook.onSpeechDown)

    def _audio_consume(self):
        while True:
            time.sleep(self.free)
            if not self.audio_queue.empty():
                data = self.audio_queue.get()
                if not isinstance(data, dict):
                    break
                mtype = data["type"]
                if mtype == "chat":
                    emotion = self.emotion.infer(data["text"])
                    run_async_func(self.hook.onConsume, data["text"], emotion, data["wav_byte"])
                elif mtype == "sing":
                    run_async_func(self.hook.onSing, data["title"], data["vocals_byte"], data["wav_byte"])
        run_async_func(self.hook.onConsumeDown)

    def sing(self, name: str):
        if self.status is not Status.ONE:
            run_async_func(self.hook.onBusy)
            return False
        title, link = get_music_mp3_url(name)
        if not link:
            run_async_func(self.hook.noSing, title)
            return False
        code = generate_unique_code(title)
        music_list = [music for music in os.listdir(self.music_dir) if music.endswith(".wav")]
        music = code + ".wav"
        data = {"title": title, "link": link, "code": code, "has": music in music_list}
        sing_generation_task = threading.Thread(target=self._sing_generation, args=(data,))
        sing_consume_task = threading.Thread(target=self._audio_consume)
        sing_consume_task.start()
        sing_generation_task.start()
        self._switch(Status.FOUR)
        run_async_func(self.hook.onStart, Action.SING)
        return True

    def __singing_voice_synthesis(self, sing: dict):
        run_async_func(self.hook.onLoadSing, sing["title"])
        download(sing["link"], f'{sing["code"]}.mp3', self.temp_dir)
        mp3_path = os.path.join(self.temp_dir, f'{sing["code"]}.mp3')
        print("音频下载成功")
        try:
            cmd = "spleeter separate -o {0} -p spleeter:2stems {1}".format(self.music_dir2, mp3_path)
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
            if result.returncode == 0:
                print("命令执行成功！")
                print("标准输出：")
                print(result.stdout)
                start = time.time()
                print("开始切片原人声音频")
                vocals_path = f'{self.music_dir2}/{sing["code"]}/vocals.wav'
                slice_wav_file(vocals_path, self.temp_dir, self.slice_num)
                print("开始切片原伴奏音频")
                accompaniment_path = f'{self.music_dir2}/{sing["code"]}/accompaniment.wav'
                slice_wav_file(accompaniment_path, self.temp2_dir, self.slice_num)
                # 待转化音频
                wav_list = [wav for wav in os.listdir(self.temp_dir) if wav.endswith(".wav")]
                wavs = [0] * len(wav_list)

                def task(wav):
                    wav_name = str(wav).replace(".wav", "")
                    accompaniment = os.path.join(self.temp2_dir, wav)
                    wav = os.path.join(self.temp_dir, wav)
                    print(f"Task {wav_name} start")
                    wav_ = self.vc.infer(wav)
                    wavs[int(wav_name)] = wav_
                    print(f"Task {wav_name} completed")
                    # 将合并后的音频保存为 WAV 文件
                    sf.write(wav, wav_, 44100, subtype='PCM_16')
                    vocals_byte = wav_to_bytes(wav)
                    combined_wav(
                        accompaniment,
                        wav,
                        wav
                    )
                    wav_byte = wav_to_bytes(wav)
                    data = {"type": "sing", "title": sing["title"], "wav_byte": wav_byte, "vocals_byte": vocals_byte}
                    self.audio_queue.put(data)

                for i in range(len(wav_list)):
                    task(f'{i}.wav')
                self.audio_queue.put(self.sign)
                print("准备合成音频")
                wav_path = os.path.join(self.temp_dir, f'{sing["code"]}.wav')
                concat_wav_slices(wavs, wav_path)
                print("合成音频成功")
                combined_wav(
                    f'{self.music_dir2}/{sing["code"]}/accompaniment.wav',
                    wav_path,
                    os.path.join(self.music_dir, f'{sing["code"]}.wav')
                )
                print("speed time {}s".format(time.time() - start))
            else:
                print("命令执行失败！")
                print("标准输出：")
                print(result.stdout)
                print("错误输出：")
                print(result.stderr)
        except Exception as e:
            print("执行命令时出现异常：", e)
        finally:
            self.__clear_temp_dir()

    def __slice_send_music(self, sing: dict):
        vocals = os.path.join("{}/{}/".format(self.music_dir2, sing["code"]), "vocals.wav")
        slice_wav_file(vocals, self.temp_dir, self.slice_num)
        wav = os.path.join(self.music_dir, f'{sing["code"]}.wav')
        length = slice_wav_file(wav, self.temp2_dir, self.slice_num)
        for i in range(length):
            name = f'{i}.wav'
            vocals = os.path.join(self.temp_dir, name)
            vocals_byte = wav_to_bytes(vocals)
            wav = os.path.join(self.temp2_dir, name)
            wav_byte = wav_to_bytes(wav)
            data = {"type": "sing", "title": sing["title"], "wav_byte": wav_byte, "vocals_byte": vocals_byte}
            self.audio_queue.put(data)
        self.audio_queue.put(self.sign)
        self.__clear_temp_dir()

    def _sing_generation(self, sing: dict):
        self.__slice_send_music(sing) if sing["has"] else self.__singing_voice_synthesis(sing)

    def __clear_temp_dir(self):
        temp_list = os.listdir(self.temp_dir)
        for temp in temp_list:
            os.remove(os.path.join(self.temp_dir, temp))
        temp2_list = os.listdir(self.temp2_dir)
        for temp in temp2_list:
            os.remove(os.path.join(self.temp2_dir, temp))
