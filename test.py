import asyncio
import websockets

async def websocket_handler(websocket, path):
    # 这个函数将处理WebSocket连接
    async for message in websocket:
        # 处理接收到的消息
        print(f"Received message: {message}")

async def start_websocket_server():
    # 启动WebSocket服务器
    server = await websockets.serve(websocket_handler, 'localhost', 8765)
    await server.wait_closed()

if __name__ == "__main__":
    # 创建一个事件循环并启动WebSocket服务器
    loop = asyncio.get_event_loop()
    loop.run_until_complete(start_websocket_server())
    loop.run_forever()
