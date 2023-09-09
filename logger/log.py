import logging
import os
from datetime import datetime

def setup_logger(log_dir):
    # 创建日志记录器
    logger = logging.getLogger("cocoBot")
    logger.setLevel(logging.INFO)

    # 创建日志格式化器
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # 获取当前日期
    current_date = datetime.now().strftime("%Y-%m-%d")

    # 创建文件处理器
    log_file = os.path.join(log_dir, f"{current_date}.log")
    file_handler = logging.FileHandler(log_file,encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # 将处理器添加到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# 创建日志目录（如果不存在）
log_directory = "logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

# 创建全局唯一的日志对象
logger = setup_logger(log_directory)
