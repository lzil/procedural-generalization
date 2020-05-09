import numpy as np
import pickle
import argparse
import sys
import os

import cv2

from helpers.utils import get_id

vid_folder = 'videos/demos'
os.makedirs(vid_folder, exists_ok=True)

# send all the output from cv2 to a log file
with open('log_visualize_demo.log', 'a') as f:
    sys.stdout = f

# only argument is demo path
parser = argparse.ArgumentParser()
parser.add_argument('demo_path')
args = parser.parse_args()
path = args.demo_path

with open(path, 'rb') as f:
    demo = pickle.load(f)

demo_id = get_id(path)
observations = demo['observations']

# set up video writer
videodims = (64, 64)
fourcc = cv2.VideoWriter_fourcc(*'avc1')
# fourcc = cv2.VideoWriter_fourcc('H', '2', '6', '4')
video = cv2.VideoWriter(os.path.join(vid_folder, f'demo_{demo_id}.mp4', fourcc, 40, videodims))

# produce video
for i in range(0, len(observations)):
    img = observations[i]
    for j in range(6):
        video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

video.release()
cv2.destroyAllWindows()
