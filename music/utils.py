# 下载文件块大小（每个部分的大小）
import asyncio
import hashlib
import os
import wave

import requests
import soundfile as sf

import aiohttp
import math

import numpy as np
from pydub import AudioSegment
from tqdm import tqdm

CHUNK_SIZE = 1024 * 1024  # 1MB

PART_NUM = 10


# 下载完成回调函数示例
def _download_part_complete(file_path):
    pass


# 异步下载文件部分
async def _download_part(session, url, start, end, file_path, pbar):
    if os.path.exists(file_path):
        # 189262416
        file_size = os.path.getsize(file_path)
        start += file_size
        pbar.update(file_size)
        if start - 1 == end:
            return
    headers = {'Range': f'bytes={start}-{end}'}
    async with session.get(url, headers=headers) as response:
        with open(file_path, 'ab') as file:
            while True:
                chunk = await response.content.read(CHUNK_SIZE)
                if not chunk:
                    break
                file.write(chunk)
                pbar.update(len(chunk))
    _download_part_complete(file_path)


async def _download_all(session, url, file_path):
    async with session.head(url) as response:
        total_size = int(response.headers.get('content-length', 0))  # 73298440
        pbar = tqdm(total=total_size, unit='B', unit_scale=True)
    start = 0
    end = total_size - 1
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        start += file_size
        pbar.update(file_size)
        if start - 1 == end:
            return
    headers = {'Range': f'bytes={start}-{end}'}
    async with session.get(url, headers=headers) as response:
        with open(file_path, 'ab') as file:
            while True:
                chunk = await response.content.read(CHUNK_SIZE)
                if not chunk:
                    break
                file.write(chunk)
                pbar.update(len(chunk))
            pbar.close()
    _download_part_complete(file_path)


def _copy_large_file(source_file, destination_file, chunk_size=8192):
    with open(source_file, "rb") as src_file, open(destination_file, "ab") as dest_file:
        while True:
            chunk = src_file.read(chunk_size)
            if not chunk:
                break
            dest_file.write(chunk)


async def fast_download(url: str, file_name=None, out_dir=None, callback=None):
    try:
        if not file_name:
            file_name = os.path.basename(url)
        if not out_dir:
            out_dir = os.path.dirname(os.path.abspath(__file__))
        if out_dir != "" and not os.path.exists(out_dir):
            os.mkdir(out_dir)
        file_path = os.path.join(out_dir, file_name)
        if os.path.exists(file_path):
            return
        async with aiohttp.ClientSession() as session:
            async with session.head(url) as response:
                total_size = int(response.headers.get('content-length', 0))  # 73298440
                pbar = tqdm(total=total_size, unit='B', unit_scale=True)
            # 开始切割
            parts_size = math.ceil(total_size / PART_NUM)
            tasks = []
            start = 0
            for i in range(PART_NUM):
                # 0 - 3664921
                end = start + parts_size
                if end > total_size - 1:
                    end = total_size - 1
                part_file_path = f"{file_path}.part{i + 1}"
                task = asyncio.create_task(_download_part(session, url, start, end, part_file_path, pbar))
                tasks.append(task)
                start = end + 1
            await asyncio.gather(*tasks)
        # 合并文件部分
        for i in range(PART_NUM):
            part_file_path = f"{file_path}.part{i + 1}"
            _copy_large_file(part_file_path, file_path)
            os.remove(part_file_path)
        pbar.close()
        if callback:
            callback()
    except asyncio.CancelledError:
        if tasks and len(tasks) > 0:
            for task in tasks:
                task.cancel()
        print("Task is cancelled.")


def download(url: str, file_name=None, out_dir=None):
    if not file_name:
        file_name = os.path.basename(url)
    if not out_dir:
        out_dir = os.path.dirname(os.path.abspath(__file__))
    if out_dir != "" and not os.path.exists(out_dir):
        os.mkdir(out_dir)
    file_path = os.path.join(out_dir, file_name)
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))  # 73298440
    pbar = tqdm(total=total_size, unit='B', unit_scale=True)
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


async def async_download(url: str, file_name=None, out_dir=None, callback=None):
    try:
        if not file_name:
            file_name = os.path.basename(url)
        if not out_dir:
            out_dir = os.path.dirname(os.path.abspath(__file__))
        if out_dir != "" and not os.path.exists(out_dir):
            os.mkdir(out_dir)
        file_path = os.path.join(out_dir, file_name)
        async with aiohttp.ClientSession() as session:
            task = asyncio.create_task(_download_all(session, url, file_path))
            await asyncio.gather(task)
        if callback:
            callback()
    except asyncio.CancelledError:
        task.cancel()
        print("Task is cancelled.")


def wav_to_bytes(wav_file):
    with open(wav_file, 'rb') as f:
        wav_bytes = f.read()
    return wav_bytes


def _fade_in_out(audio, fade_length):
    fade_in = np.linspace(0, 1, fade_length)
    fade_out = np.linspace(1, 0, fade_length)
    audio[:fade_length] *= fade_in
    audio[-fade_length:] *= fade_out
    return audio


def concat_wav_slices(audio_pieces: list, output_file, fade_length=4410):
    # 连接所有音频片段
    audio_concatenated = np.concatenate(audio_pieces)

    # 对连接后的音频进行淡入淡出处理，实现平滑连接
    audio_concatenated = _fade_in_out(audio_concatenated, fade_length)

    # 将合并后的音频保存为 WAV 文件
    sf.write(output_file, audio_concatenated, 44100, subtype='PCM_16')

    print("WAV slices have been concatenated into a single WAV file:", output_file)


def slice_wav_file(input_file, output_dir, slice_length):
    i = 0
    with wave.open(input_file, 'rb') as wav_file:
        params = wav_file.getparams()
        frame_rate = params.framerate
        num_frames = params.nframes

        slice_samples = int(slice_length * frame_rate)
        num_slices = int(np.ceil(num_frames / slice_samples))

        for i in range(num_slices):
            start_frame = i * slice_samples
            end_frame = min(start_frame + slice_samples, num_frames)

            wav_file.setpos(start_frame)
            frames = wav_file.readframes(end_frame - start_frame)

            output_file = os.path.join(output_dir, "{}.wav".format(i))
            with wave.open(output_file, 'wb') as slice_wav:
                slice_wav.setparams(params)
                slice_wav.writeframes(frames)
                i += 1
        return i


def generate_unique_code(input_string):
    # 使用MD5哈希函数计算哈希值，并转换为16进制表示
    hash_object = hashlib.md5(input_string.encode())
    unique_code = hash_object.hexdigest()
    return unique_code

if __name__ == '__main__':
    # https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin
    try:
        asyncio.run(download(url="https://allall02.baidupcs.com/file/5a9c8c125m360631f0e0bb9082098601?bkt=en-2e2b5030dd6ff037d2f710cfcf14cf19f147d8b01e62ebad22df1c4912584c219d0eb88c6a18b956&fid=928833932-250528-739339719529079&time=1691913499&sign=FDTAXUbGERLQlBHSKfWqi-DCb740ccc5511e5e8fedcff06b081203-wbhQPCZv8Tr7FDhpnMBz79xpX6I%3D&to=80&size=36278875&sta_dx=36278875&sta_cs=2914&sta_ft=zip&sta_ct=7&sta_mt=0&fm2=MH%2CYangquan%2CAnywhere%2C%2C%E6%B1%9F%E8%A5%BF%2Ccmnet&ctime=1650776796&mtime=1691913451&resv0=-1&resv1=0&resv2=rlim&resv3=5&resv4=36278875&vuk=928833932&iv=0&htype=&randtype=&tkbind_id=0&newver=1&newfm=1&secfm=1&flow_ver=3&pkey=en-f9cdfcfedd9807e2088c298c17c62d089462c62059eec784abd25948040a77e68cf3d241174611c0&sl=76480590&expires=8h&rt=pr&r=363100165&vbdid=3069477014&fin=%E9%BB%91%E9%A9%AC%E7%A8%8B%E5%BA%8F%E5%91%98%E5%8C%A0%E5%BF%83%E4%B9%8B%E4%BD%9CC%2B%2B%E6%95%99%E7%A8%8B%E4%BB%8E0%E5%88%B01%E5%85%A5%E9%97%A8%E7%BC%96%E7%A8%8B.zip&fn=%E9%BB%91%E9%A9%AC%E7%A8%8B%E5%BA%8F%E5%91%98%E5%8C%A0%E5%BF%83%E4%B9%8B%E4%BD%9CC%2B%2B%E6%95%99%E7%A8%8B%E4%BB%8E0%E5%88%B01%E5%85%A5%E9%97%A8%E7%BC%96%E7%A8%8B.zip&rtype=1&dp-logid=8669391782205887089&dp-callid=0.1&hps=1&tsl=80&csl=80&fsl=-1&csign=kv1Xm5yu5JxerX%2B0vI8mrE57CKc%3D&so=0&ut=6&uter=4&serv=0&uc=3918998975&ti=6c32122c2edef1eb3e2745c3910b7d645e2e44b7a45919e3305a5e1275657320&hflag=30&from_type=3&adg=c_ac87836267e44135302522553687f12f&reqlabel=250528_f_105267947e6099021b1cdf787877d45e_-1_bcefc6e3e9298fc5db673915b0bdd817&by=themis"))
    except KeyboardInterrupt:
        print("取消下载成功")
