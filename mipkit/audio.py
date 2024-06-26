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

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import librosa
    import numpy as np
    import pylab
    from matplotlib import cm

# try:
#     import librosa
#     import numpy as np
#     import pylab
#     from matplotlib import cm
# except ImportError as e:
#     warnings.warn(e.msg)


def load_file(fp, sr=None):
    X, sr = librosa.load(fp, sr=sr)
    return X, sr


def scale_minmax(X, min=0.0, max=1.0):
    esp = 10e-9
    X_std = (X - X.min()) / (X.max() - X.min() + esp)
    X_scaled = X_std * (max - min) + min
    return X_scaled


# def spectrogram_image_from_file(
#     wav_path, path_to_save=None, sr=16000, hop_length=128, n_mels=256, to_3c=True
# ):
#     data = load_sound_file(wav_path, sr=sr)
#     return spectrogram_image(data, path_to_save, sr, hop_length, n_mels, to_3c)


def image_2to3_channels(img):
    return np.dstack([img, img, img])


N_FFT = 1024  # 2048
HOP_LENGTH = 512
N_MELS = 96
N_MFCC = 40


def wav2mfcc(wav_arr, sr, n_mfcc=N_FFT, **args):
    mfcc = librosa.feature.mfcc(y=wav_arr, sr=sr, S=None, n_mfcc=n_mfcc, **args)
    return mfcc


def save_mfcc(mel, path_to_save):
    # Plotting the spectrogram and save as JPG without axes (just the image)
    # pylab.figure(figsize=(3, 3))
    pylab.axis("off")
    pylab.axes([0.0, 0.0, 1.0, 1.0], frameon=False, xticks=[], yticks=[])  # Remove the white edge
    librosa.display.specshow(mel, cmap=cm.jet)
    pylab.savefig(path_to_save, bbox_inches=None, pad_inches=0)
    pylab.close()


def wav2mel(wav_arr, sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    S = librosa.feature.melspectrogram(
        wav_arr, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    log_S = librosa.power_to_db(S, ref=np.max)
    return log_S


def save_mel(mel, path_to_save):
    # Plotting the spectrogram and save as JPG without axes (just the image)
    # pylab.figure(figsize=(3, 3))
    pylab.axis("off")
    pylab.axes([0.0, 0.0, 1.0, 1.0], frameon=False, xticks=[], yticks=[])  # Remove the white edge
    librosa.display.specshow(mel, cmap=cm.jet)
    pylab.savefig(path_to_save, bbox_inches=None, pad_inches=0)
    pylab.close()


# def spectrogram_image(
#     y,
#     n_fft,
#     path_to_save=None,
#     sr=16000,
#     hop_length=128,
#     n_mels=256,
#     to_3c=False,
# ):
#     # use log-melspectrogram
#     # mels = librosa.feature.melspectrogram(
#     #     y=y, sr=sr, n_mels=n_mels, n_fft=hop_length * 2, hop_length=hop_length
#     # )
#     mel = librosa.feature.melspectrogram(
#         y=y,
#         sr=sr,
#         n_fft=n_fft,
#         hop_length=hop_length,
#         n_mels=n_mels,
#     )

#     # mel = np.log(mel + 1e-9)  # add small number to avoid log(0)

#     mel_dB = librosa.power_to_db(mel, ref=np.max)

#     # min-max scale to fit inside 8-bit range
#     # img = scale_minmax(mel_dB, 0, 255).astype(np.uint8)
#     # img = np.flip(img, axis=0)  # put low frequencies at the bottom in image
#     # img = 255 - img  # invert. make black==more energy
#     # return mel_dB
#     # if to_3c:
#     #     img = image_2to3_channels(img)

#     if path_to_save is not None:
#         plt.axis("off")
#         plt.savefig(path_to_save)
#     else:
#         return img
