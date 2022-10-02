import cv2
import glob

def load_images_to_video(imgs_path: str, video_name: str = 'random_run.mp4'):

    files = sorted(glob.glob(imgs_path + '*'))
    f0 = cv2.imread(files[0])
    frameSize = (f0.shape[1], f0.shape[0])
    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 60, frameSize)
    for filename in files:
        img = cv2.imread(filename)
        out.write(img)
    out.release()