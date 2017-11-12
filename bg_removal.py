import cv2

import sys
import argparse as ap

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

video_width = int(video.get(3))
video_height = int(video.get(4))
video_fps = 1 / video.get(2)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')

video_name = video_file_name.split('.')[0]
out = cv2.VideoWriter('testset/' + video_name + '_bgrm.avi', fourcc, video_fps, (video_width, video_height))
