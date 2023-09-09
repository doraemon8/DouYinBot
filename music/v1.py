import requests
from urllib.parse import quote

BASE_URL = "https://www.yeyulingfeng.com/tools/music/"

data = {
    "filter": "name",
    "type": "netease",
    "page": 1
}
headers = {
    "X-Requested-With": "XMLHttpRequest",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Origin": "https://www.yeyulingfeng.com",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 Edg/114.0.1823.79"
}


def get_music_mp3_url(keyword:str):
    data["input"] = keyword
    headers["Referer"] = BASE_URL+"?name={0}&type=netease".format(quote(keyword))
    resp = requests.post(BASE_URL,data=data,headers=headers)
    _data = resp.json()
    for music in _data["data"]:
        if music["url"]:
            return music["title"], music["url"]
    return None

if __name__ == '__main__':
    print(get_music_mp3_url("天外来物"))
