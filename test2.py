from configparser import ConfigParser

from bot import CocoBot, CocoBotHook


class Hook(CocoBotHook):
    async def onConsume(self, text: str, emotion: dict, wav_byte: bytes):
        pass

    async def onSing(self, name: str, vocals_byte: bytes, wav_byte: bytes):
        pass


config = ConfigParser()
config.read('config.ini', encoding="utf-8")
chatbot = CocoBot(config)
chatbot.bind(Hook())




if __name__ == '__main__':
    chatbot.chat("你好")
