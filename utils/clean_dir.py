import glob
import os

def clean_dir(dir_path: str):
    for f in glob.glob(dir_path + '*'):
        os.remove(f)
