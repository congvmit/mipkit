"""
 The MIT License (MIT)
 Copyright (c) 2021 Cong Vo
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 Provided license texts might have their own copyrights and restrictions
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
"""
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
