import sys
import os

import math

import imageio
from moviepy.editor import *

# Read video from file
video_name = sys.argv[1] ## example: test_video_01
video_name_input = 'testset/' + video_name + '.mov'
video_input = VideoFileClip(video_name_input)

video_frame = int(video_input.duration * video_input.fps) ## duration: second / fps: frame per second

os.makedirs('testset/' + video_name)

video_frame_length = math.ceil(math.log(video_frame, 10)) ## ex. 720 -> 3

for i in range(0, video_frame):
    video_input.save_frame('testset/' + video_name + '/frame_' + str(i).zfill(video_frame_length) + '.jpg', i)
