# Copyright (c) 2021 congvm
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from pytube import YouTube
from PIL import Image
import io
import numpy as np
import urllib.request as urllib_request
from io import BytesIO
import time


def download_img_with_url(url, retry=0, retry_gap=0.1, proxy=None):
    if proxy == None:
        proxies = {}
    else:
        proxies = {'http': proxy, 'https': proxy}
    try:
        proxy_handler = urllib_request.ProxyHandler(proxies)
        opener = urllib_request.build_opener(proxy_handler)
        img = Image.open(BytesIO(opener.open(url).read())).convert('RGB')
        return img
    except Exception as e:
        if retry > 0:
            time.sleep(retry_gap)
            return download_img_with_url(url, retry=retry-1, proxy=proxy)
        else:
            return None

def download_from_youtube(output_path, video_id):
    yt = YouTube(f'https://youtu.be/watch?v={video_id}').streams.first().download(
        output_path=output_path, filename=video_id)
