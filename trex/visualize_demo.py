import numpy as np
import pickle
import argparse
import sys
import os

import cv2

from helpers.utils import get_id
from train_reward import get_file

vid_folder = 'videos'
os.makedirs(vid_folder, exist_ok=True)

# # send all the output from cv2 to a log file
# os.makedirs('logs', exist_ok=True)
# f = open('logs/log_visualize_demo.log', 'a')
# sys.stdout = f
# sys.stderr = f

# only argument is demo path
parser = argparse.ArgumentParser()
parser.add_argument('demo_id')
args = parser.parse_args()

demo = get_file(args.demo_id + '.demo')

demo_id = demo['demo_id']
observations = demo['observations']

video_path = os.path.join(vid_folder, f'demo_{demo_id}.mp4')
if os.path.isfile(video_path):
    os.remove(video_path)

# set up video writer
videodims = (64, 64)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# fourcc = cv2.VideoWriter_fourcc('H', '2', '6', '4')
video = cv2.VideoWriter(video_path, fourcc, 40, videodims)

# produce video
for i in range(0, len(observations)):
    img = observations[i]
    for j in range(6):
        video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

video.release()
cv2.destroyAllWindows()
# f.close()
