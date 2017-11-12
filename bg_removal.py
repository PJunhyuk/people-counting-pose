## default
import sys
import argparse as ap

## opencv
import cv2

## moviepy
from moviepy.editor import *

parser = ap.ArgumentParser()
parser.add_argument('-f', "--videoFile", help="Path to Video File")

args = vars(parser.parse_args())

if args["videoFile"] is not None:
    video_file_name = args["videoFile"]
else:
    print("You have to input videoFile name")
    sys.exit(1)

video_file_route = 'testset/' + video_file_name
video = cv2.VideoCapture(video_file_route)

if not video.isOpened():
    print("Video doesn't opened!")
    sys.exit(1)

video_width = int(video.get(3))
video_height = int(video.get(4))
video_fps = 1 / video.get(2)

frame_output_list = []

while(1):
    ret, frame = video.read()

    frame_output = frame

    frame_output_list.append(frame_output)

    if ret == False:
        break

video_output = ImageSequenceClip(frame_output_list, fps=video_fps)
video_name = video_file_name.split('.')[0]
video_output.write_videofile("testset/" + video_name + "_bgrm.mp4", fps=video.fps, progress_bar=False)
