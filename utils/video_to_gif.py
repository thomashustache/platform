import cv2
import glob
import os
import shutil
import PySimpleGUI as sg
from PIL import Image
from utils.clean_dir import clean_dir

file_types = [("MP4 (*.mp4)", "*.mp4"), ("All files (*.*)", "*.*")]

def convert_mp4_to_jpgs(video_path: str, jpgs_path: str):
    os.makedirs(jpgs_path, exist_ok=True)
    video_capture = cv2.VideoCapture(video_path)
    still_reading, image = video_capture.read()
    frame_count = 0
    if os.path.exists(jpgs_path):
        clean_dir(jpgs_path)
        # remove previous GIF frame files

    while still_reading:
        cv2.imwrite(jpgs_path + f"frame_{frame_count:05d}.jpg", image)

        # read next image
        still_reading, image = video_capture.read()
        frame_count += 1


def make_gif_from_video(video_path: str,
                        gif_folder: str,
                        gif_name: str,
                        jpgs_frame_folder: str,
                        skip_rate: int = 1):

    convert_mp4_to_jpgs(video_path, jpgs_frame_folder)
    images = glob.glob(f"{jpgs_frame_folder}/*.jpg")
    images.sort()
    frames = [Image.open(image) for ind, image in enumerate(images) if ind % skip_rate == 0]
    frame_one = frames[0]
    frame_one.save(gif_folder + gif_name, format="GIF", append_images=frames,
                   save_all=True, duration=1, loop=1)


if __name__ == '__main__':

    video_path = 'results/videos/QPAMDP/fixed_stds/optimstep_0_reward-0.247.mp4'
    jpgs_path = 'results/jpgs/'
    gif_folder = 'results/gifs/QPAMDP/fixed_stds/'
    gif_name = 'optimstep_0_reward-0.247.gif'

    from moviepy.editor import *
    clip = (VideoFileClip(video_path))
    os.makedirs(gif_folder, exist_ok=True)
    clip.write_gif(gif_folder + gif_name)

    make_gif_from_video(video_path, gif_folder, gif_name, jpgs_path)

