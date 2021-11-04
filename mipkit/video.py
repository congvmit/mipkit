import cv2
from typing import Tuple
import numpy as np


def load_video(video_file: str) -> Tuple[cv2.VideoCapture, Tuple[int, int], int, int]:
    """Load a video from the specified file.

    Args:
        video_file (str): a path to video file.

    Returns:
        VideoCapture: a VideoCapture object.
        (width, height): width and height of the video.
        num_frames: number of frames
        fps: Frame Per Second
    """
    cap = cv2.VideoCapture(video_file)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    return cap, (width, height), num_frames, fps


def get_frame_by_idx(cap: cv2.VideoCapture, idx='middle'):
    if not isinstance(idx, int):
        assert idx in ['middle', 'first', 'last']

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if idx == 'middle':
        idx = num_frames // 2
    elif idx == 'first':
        idx = 0
    elif idx == 'last':
        idx = num_frames - 1
    if idx > num_frames:
        raise ValueError('`frame_idx` must be less than a number of frames')

    return_frames = None

    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    has_frame, return_frames = cap.read()
    # Convert from BGR to RGB
    return_frames = cv2.cvtColor(return_frames, cv2.COLOR_BGR2RGB)
    return return_frames


def get_frames_by_FPS(cap: cv2.VideoCapture, fps=1):
    """Get frames from video given by FPS and

    Args:
        cap (cv2.VideoCapture): VideoCapture instance
        fps (int, optional): Frame per sec. Defaults to 1.

    Returns:
        list_frames (list): List of frames
    """
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return_frames = []
    frame_indexes = range(0, num_frames, fps)
    for findex in frame_indexes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, findex)
        has_frame, frame = cap.read()

        # Convert from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return_frames.append(frame)
    return return_frames
