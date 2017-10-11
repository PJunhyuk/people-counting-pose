## Import the required modules
# Check time required
import time
time_start = time.clock()

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
import cv2

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
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-v', "--videoFile", help="Path to Video File")
args = vars(parser.parse_args())
video_name = args["videoFile"]

video = video_pose.read_video(video_name)

video_size_x = video.size[0] # type: int
video_size_y = video.size[1] # type: int

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

##########

# for object-tracker
target_points = [] # format: [(minx, miny, maxx, maxy), (minx, miny, maxx, maxy) ... ]
tracker = []

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
    print('people_num: ' + str(people_num))

    #####

    if i != 0:
        for k in range(len(tracker)):
            tracker[k].update(image)
            rect = tracker[k].get_position()
            if int(rect.left()) <= 0 or int(rect.top()) <= 0 or int(rect.right()) >= video_size_x or int(rect.bottom()) >= video_size_y:
                # object left(leave)
                print('Object GONE!')
                del tracker[k]
            else:
                draw.rectangle([rect.left(), rect.top(), rect.right(), rect.bottom()], outline='red', outline=5)
                print('Object ' + str(k) + ' tracked at [' + str(int(rect.left())) + ',' + str(int(rect.top())) + ', ' + str(int(rect.right())) + ',' + str(int(rect.bottom())) + ']')

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
                    tracking_people_count = tracking_people_count + 1

    ##########

    if i == 0:
        # Initial co-ordinates of the object to be tracked
        # Create the tracker object
        tracker = [dlib.correlation_tracker() for _ in range(len(target_points))]
        # Provide the tracker the initial position of the object
        [tracker[i].start_track(image, dlib.rectangle(*rect)) for i, rect in enumerate(target_points)]
        tracking_people_count = int(len(tracker))

    #####

    draw.text((0, 0), 'People(this frame, by pose): ' + str(people_real_num) + ' (threshold = ' + str(point_min) + ')', (0,0,0), font=font)
    draw.text((0, 18), 'People(cumulative, by tracking): ' + str(tracking_people_count), (0,0,0), font=font)
    draw.text((0, 36), 'Frame: ' + str(i) + '/' + str(video_frame_number), (0,0,0), font=font)
    draw.text((0, 54), 'Total time required: ' + str(round(time.clock() - time_start, 1)) + 'sec', (0,0,0), font=font)

    print('people_real_num: ' + str(people_real_num))
    print('frame: ' + str(i))

    print('len(target_points): ' + str(len(target_points)))

    image_img_numpy = np.asarray(image_img)

    pose_frame_list.append(image_img_numpy)

video_pose = ImageSequenceClip(pose_frame_list, fps=video.fps)
video_pose.write_videofile("testset/" + video_name + "_tracking.mp4", fps=video.fps)

print("Time(s): " + str(time.clock() - time_start))
