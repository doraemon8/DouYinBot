import asyncio
import os
import threading
from abc import ABC, abstractmethod

import websockets

from .driver import DriverClass, DriverFactory, Driver
from .setting import Setting,root


class Barrage(threading.Thread):
    def __init__(self):
        super().__init__()
        self.page: str | None = None
        self.host = "127.0.0.1"
        self.port: int | None = None
        self.isWss = False
        self.driver = None
        self.callback = None
        self.ws = None

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        if not self.driver:
            self.driver = DriverFactory.create(DriverClass.CHROME)
        wsurl = f"ws://{self.host}:{self.port}"
        if self.isWss:
            wsurl =f"wss://{self.host}/{self.port}"
        script = os.path.join(root,"script/douyin.js")
        self.driver.inject(self.page, script, Setting(wsurl))
        print(f"服务启动{wsurl}")
        loop.run_until_complete(self._start_websocket_server())
        loop.run_forever()
        print(f"服务关闭{wsurl}")

    async def _start_websocket_server(self):
        # 启动WebSocket服务器
        self.ws = await websockets.serve(self.__handler, self.host, self.port)
        await self.ws.wait_closed()

    async def __handler(self, websocket, path):
        # 这个函数将处理WebSocket连接
        async for message in websocket:
            # 处理接收到的消息
            await self.callback(message)


class Builder(ABC):
    @abstractmethod
    def port(self, port: int):
        pass

    @abstractmethod
    def host(self, host: str):
        pass

    @abstractmethod
    def driver(self, driver: DriverClass):
        pass

    @abstractmethod
    def page(self, url: str):
        pass

    @abstractmethod
    def enableWSS(self):
        pass

    @abstractmethod
    def on(self, callback):
        pass


class DYBarrageBuilder(Builder, ABC):
    def __init__(self):
        self.barrage = Barrage()

    def host(self, host: str):
        self.barrage.host = host
        return self


    def page(self, url: str):
        self.barrage.page = url
        return self

    def port(self, port: int):
        self.barrage.port = port
        return self

    def driver(self, driver: DriverClass):
        self.barrage.driver = DriverFactory.create(driver)
        return self

    def on(self, callback):
        self.barrage.callback = callback
        return self

    def enableWSS(self):
        self.barrage.isWss = True
        return self

    def build(self) -> Barrage:
        return self.barrage
