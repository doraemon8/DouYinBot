import asyncio

from barrage import DYBarrageBuilder

url = "https://live.douyin.com/639709145929"


def callback(message):
    print(message)




if __name__ == '__main__':
    barrage = DYBarrageBuilder().port(8080).page(url).on(callback).build()
    barrage.daemon=True
    barrage.start()
