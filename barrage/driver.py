import os
from abc import ABC, abstractmethod
from enum import Enum
import platform
from selenium import webdriver

from .setting import Setting

current_os = platform.system()

if current_os == "Windows":
    # 在Windows中设置环境变量
    os.environ['MY_VARIABLE'] = 'C:\\Users\\chromedriver.exe'
elif current_os == "Linux":
    # 在linux中设置环境变量
    os.environ['MY_VARIABLE'] = '/root/chromedriver.exe'


class DriverClass(Enum):
    CHROME = "谷歌驱动"


class Driver(ABC):

    def __init__(self) -> None:
        self.driver = None
        self.event = {}

    @abstractmethod
    def inject(self, url: str,script:str, options:Setting):
        pass



    def close(self):
        if self.driver:
            self.driver.quit()


class ChromeDriver(Driver):

    def inject(self, url: str,script:str, setting=Setting()):
        self.driver = webdriver.Chrome()
        self.driver.get(url)
        options = setting.to_dict()
        with open(script, "r", encoding="utf-8") as f:
            self.driver.execute_script( f.read(), options.get("wsurl"), options.get("timeinterval"))




class DriverFactory:

    @staticmethod
    def create(driver: DriverClass) -> Driver:
        if driver.CHROME:
            return ChromeDriver()
