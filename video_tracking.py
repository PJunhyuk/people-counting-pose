## Import the required modules
# Check time required
import time
time_start = time.time()

import sys
import os
import argparse as ap

import math

import imageio
from moviepy.editor import *

import numpy as np

sys.path.append(os.path.dirname(__file__) + "/../")

from scipy.misc import imread, imsave

from config import load_config
from dataset.factory import create as create_dataset
from nnet import predict
from util import visualize
from dataset.pose_dataset import data_to_input

from multiperson.detections import extract_detections
from multiperson.predict import SpatialModel, eval_graph, get_person_conf_multicut
from multiperson.visualize import PersonDraw, visualize_detections

import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont
font = ImageFont.truetype("./font/NotoSans-Bold.ttf", 12)

import random

# for object-tracker
import dlib

import video_pose

####################

cfg = load_config("demo/pose_cfg_multi.yaml")

dataset = create_dataset(cfg)

sm = SpatialModel(cfg)
sm.load()

draw_multi = PersonDraw()

# Load and setup CNN part detector
sess, inputs, outputs = predict.setup_pose_prediction(cfg)

##########
## Get the source of video

parser = ap.ArgumentParser()
parser.add_argument('-f', "--videoFile", help="Path to Video File")
parser.add_argument('-w', "--videoWidth", help="Width of Output Video")
parser.add_argument('-o', "--videoType", help="Extension of Output Video")

args = vars(parser.parse_args())

if args["videoFile"] is not None:
    video_name = args["videoFile"]
else:
    print("You have to input videoFile name")
    sys.exit(1)
video_output_name = video_name.split('.')[0]
video = video_pose.read_video(video_name)
print("Input video size: [" + str(video.size[0]) + ", " + str(video.size[1]) + "]")

if args["videoWidth"] is not None:
    video_width = int(args["videoWidth"])
    video = video.resize(width = video_width)
print("Changed video size: [" + str(video.size[0]) + ", " + str(video.size[1]) + "]")

if args["videoType"] is not None:
    video_type = args["videoType"]
else:
    video_type = "mp4"
print("Output video type: " + video_type)

##########
## Define some functions to mark at image

def ellipse_set(person_conf_multi, people_i, point_i):
    return (person_conf_multi[people_i][point_i][0] - point_r, person_conf_multi[people_i][point_i][1] - point_r, person_conf_multi[people_i][point_i][0] + point_r, person_conf_multi[people_i][point_i][1] + point_r)

##########

video_frame_number = int(video.duration * video.fps) ## duration: second / fps: frame per second
video_frame_ciphers = math.ceil(math.log(video_frame_number, 10)) ## ex. 720 -> 3

pose_frame_list = []

point_r = 3 # radius of points
point_min = 14 # threshold of points - If there are more than point_min points in person, we define he/she is REAL PERSON
point_num = 17 # There are 17 points in 1 person

tracking_people_count = 0
tracker_len_prev = 0

##########

# for object-tracker
target_points = [] # format: [(minx, miny, maxx, maxy), (minx, miny, maxx, maxy) ... ]
tracker = []

if not (os.path.isdir("testset/" + video_output_name)):
    os.mkdir("testset/" + video_output_name)

for i in range(0, video_frame_number):
    # Save i-th frame as image
    image = video.get_frame(i/video.fps)

    ##########
    ## By pose-tensorflow

    image_batch = data_to_input(image)

    # Compute prediction with the CNN
    outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
    scmap, locref, pairwise_diff = predict.extract_cnn_output(outputs_np, cfg, dataset.pairwise_stats)

    detections = extract_detections(cfg, scmap, locref, pairwise_diff)
    unLab, pos_array, unary_array, pwidx_array, pw_array = eval_graph(sm, detections)
    person_conf_multi = get_person_conf_multicut(sm, unLab, unary_array, pos_array)

    #####

    # Add library to draw image
    image_img = Image.fromarray(image)

    # Prepare saving image with points of pose
    draw = ImageDraw.Draw(image_img)

    #####

    people_num = 0
    people_real_num = 0

    people_num = person_conf_multi.size / (point_num * 2)
    people_num = int(people_num)

    #####

    if i != 0:
        tracker_left = []
        for k in range(len(tracker)):
            tracker[k].update(image)
            rect = tracker[k].get_position()
            if int(rect.left()) <= 0 or int(rect.top()) <= 0 or int(rect.right()) >= video.size[0] or int(rect.bottom()) >= video.size[1]:
                # object left(leave)
                print('Object GONE!')
                tracker_left.append(k)
            else:
                draw.rectangle([rect.left(), rect.top(), rect.right(), rect.bottom()], outline='red')
                print('Object ' + str(k) + ' tracked at [' + str(int(rect.left())) + ',' + str(int(rect.top())) + ', ' + str(int(rect.right())) + ',' + str(int(rect.bottom())) + ']')
        if len(tracker_left) != 0:
            for j in range(len(tracker_left)):
                del tracker[tracker_left[len(tracker_left) - 1 - j]]

    #####

    for people_i in range(0, people_num):
        point_color_r = random.randrange(0, 256)
        point_color_g = random.randrange(0, 256)
        point_color_b = random.randrange(0, 256)
        point_color = (point_color_r, point_color_g, point_color_b, 255)
        point_list = []
        point_count = 0
        point_i = 0 # index of points

        # To find rectangle which include that people - list of points x, y coordinates
        people_x = []
        people_y = []

        for point_i in range(0, point_num):
            if person_conf_multi[people_i][point_i][0] + person_conf_multi[people_i][point_i][1] != 0: # If coordinates of point is (0, 0) == meaningless data
                point_count = point_count + 1
                point_list.append(point_i)

        if point_count >= point_min:
            people_real_num = people_real_num + 1
            for point_i in range(0, point_num):
                if person_conf_multi[people_i][point_i][0] + person_conf_multi[people_i][point_i][1] != 0: # If coordinates of point is (0, 0) == meaningless data
                    draw.ellipse(ellipse_set(person_conf_multi, people_i, point_i), fill=point_color)
                    people_x.append(person_conf_multi[people_i][point_i][0])
                    people_y.append(person_conf_multi[people_i][point_i][1])
            if i == 0:
                target_points.append((int(min(people_x)), int(min(people_y)), int(max(people_x)), int(max(people_y))))
            else:
                is_new_person = True
                for k in range(len(tracker)):
                    rect = tracker[k].get_position()
                    if np.mean(people_x) < rect.right() and np.mean(people_x) > rect.left() and np.mean(people_y) < rect.bottom() and np.mean(people_y) > rect.top():
                        is_new_person = False
                if is_new_person == True:
                    tracker.append(dlib.correlation_tracker())
                    print('is_new_person!')
                    rect_temp = []
                    rect_temp.append((int(min(people_x)), int(min(people_y)), int(max(people_x)), int(max(people_y))))
                    [tracker[i+len(tracker)-1].start_track(image, dlib.rectangle(*rect)) for i, rect in enumerate(rect_temp)]

    ##########

    if i == 0:
        # Initial co-ordinates of the object to be tracked
        # Create the tracker object
        tracker = [dlib.correlation_tracker() for _ in range(len(target_points))]
        # Provide the tracker the initial position of the object
        [tracker[i].start_track(image, dlib.rectangle(*rect)) for i, rect in enumerate(target_points)]

    #####

    if tracker_len_prev < int(len(tracker)):
        tracking_people_count = tracking_people_count + int(len(tracker)) - tracker_len_prev
    tracker_len_prev = int(len(tracker))

    draw.text((0, 0), 'People(this frame): ' + str(len(tracker)), (0,0,0), font=font)
    draw.text((0, 18), 'People(cumulative): ' + str(tracking_people_count), (0,0,0), font=font)
    draw.text((0, 36), 'Frame: ' + str(i) + '/' + str(video_frame_number), (0,0,0), font=font)
    draw.text((0, 54), 'Total time required: ' + str(round(time.time() - time_start, 1)) + 'sec', (0,0,0), font=font)

    print('People(this frame): ' + str(len(tracker)))
    print('People(cumulative): ' + str(tracking_people_count))
    print('Frame: ' + str(i) + "/" + str(video_frame_number))
    print('Time required: ' + str(round(time.time() - time_start, 1)) + 'sec')

    image_img_numpy = np.asarray(image_img)

    pose_frame_list.append(image_img_numpy)

    image_name = "testset/" + video_output_name + "/" + str(i) + "_" + str(video.fps) + "_" + str(tracking_people_count) + ".jpg"
    print(image_name)
    image_img.save(image_name, "JPG")

video_pose = ImageSequenceClip(pose_frame_list, fps=video.fps)
video_pose.write_videofile("testset/" + video_output_name + "_tracking." + video_type, fps=video.fps, progress_bar=False)

print("Time(s): " + str(time.time() - time_start))
print("Output video size: [" + str(video.size[0]) + ", " + str(video.size[1]) + "]")
