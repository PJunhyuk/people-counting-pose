import sys

import imageio
from moviepy.editor import *

# Read video from file
video_name = sys.argv[1] ## example: test_video_01
video_name_input = 'testset/' + file_name + '.mov'
video_input = VideoFileClip(file_name_input)

video_frame = video_input.duration * video_input.fps ## duration: second / fps: frame per second

for i in range(0, video_frame):
    video_input.save_frame('testset/' + video_name + '/frame_' + str(i) + '.jpg', i)
