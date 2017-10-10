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
font = ImageFont.truetype("./font/NotoSans-Bold.ttf", 24)

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

parser = ap.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-v', "--videoFile", help="Path to Video File")
args = vars(parser.parse_args())
video_name = args["videoFile"]

# Get the source of video
video = video_pose.read_video(video_name)

##########

video_frame_number = int(video.duration * video.fps) ## duration: second / fps: frame per second
video_frame_ciphers = math.ceil(math.log(video_frame_number, 10)) ## ex. 720 -> 3

pose_frame_list = []

point_r = 3 # radius of points
point_min = 10 # threshold of points - If there are more than point_min points in person, we define he/she is REAL PERSON
part_min = 3 # threshold of parts - If there are more than part_min parts in person, we define he/she is REAL PERSON / part means head, arm and leg
point_num = 17 # There are 17 points in 1 person

def ellipse_set(person_conf_multi, people_i, point_i):
    return (person_conf_multi[people_i][point_i][0] - point_r, person_conf_multi[people_i][point_i][1] - point_r, person_conf_multi[people_i][point_i][0] + point_r, person_conf_multi[people_i][point_i][1] + point_r)

def line_set(person_conf_multi, people_i, point_i, point_j):
    return (person_conf_multi[people_i][point_i][0], person_conf_multi[people_i][point_i][1], person_conf_multi[people_i][point_j][0], person_conf_multi[people_i][point_j][1])

def draw_ellipse_and_line(draw, person_conf_multi, people_i, a, b, c, point_color):
    draw.ellipse(ellipse_set(person_conf_multi, people_i, a), fill=point_color)
    draw.ellipse(ellipse_set(person_conf_multi, people_i, b), fill=point_color)
    draw.ellipse(ellipse_set(person_conf_multi, people_i, c), fill=point_color)
    draw.line(line_set(person_conf_multi, people_i, a, b), fill=point_color, width=5)
    draw.line(line_set(person_conf_multi, people_i, b, c), fill=point_color, width=5)

for i in range(0, video_frame_number):
    image = video.get_frame(i/video.fps)

    ######################

    image_batch = data_to_input(image)

    # Compute prediction with the CNN
    outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
    scmap, locref, pairwise_diff = predict.extract_cnn_output(outputs_np, cfg, dataset.pairwise_stats)

    detections = extract_detections(cfg, scmap, locref, pairwise_diff)
    unLab, pos_array, unary_array, pwidx_array, pw_array = eval_graph(sm, detections)
    person_conf_multi = get_person_conf_multicut(sm, unLab, unary_array, pos_array)

    # print('person_conf_multi: ')
    # print(type(person_conf_multi))
    # print(person_conf_multi)

    # Add library to save image
    image_img = Image.fromarray(image)

    # Save image with points of pose
    draw = ImageDraw.Draw(image_img)

    people_num = 0
    people_real_num = 0
    people_part_num = 0

    people_num = person_conf_multi.size / (point_num * 2)
    people_num = int(people_num)
    print('people_num: ' + str(people_num))

    # for object-tracker
    target_points = [] # format: [(minx, miny, maxx, maxy), (minx, miny, maxx, maxy) ... ]

    for people_i in range(0, people_num):
        point_color_r = random.randrange(0, 256)
        point_color_g = random.randrange(0, 256)
        point_color_b = random.randrange(0, 256)
        point_color = (point_color_r, point_color_g, point_color_b, 255)
        point_list = []
        point_count = 0
        point_i = 0 # index of points
        part_count = 0 # count of parts in THAT person

        # To find rectangle which include that people - list of points x, y coordinates
        people_x = []
        people_y = []

        for point_i in range(0, point_num):
            if person_conf_multi[people_i][point_i][0] + person_conf_multi[people_i][point_i][1] != 0: # If coordinates of point is (0, 0) == meaningless data
                point_count = point_count + 1
                point_list.append(point_i)

        # Draw each parts
        if (5 in point_list) and (7 in point_list) and (9 in point_list): # Draw left arm
            draw_ellipse_and_line(draw, person_conf_multi, people_i, 5, 7, 9, point_color)
            part_count = part_count + 1
        if (6 in point_list) and (8 in point_list) and (10 in point_list): # Draw right arm
            draw_ellipse_and_line(draw, person_conf_multi, people_i, 6, 8, 10, point_color)
            part_count = part_count + 1
        if (11 in point_list) and (13 in point_list) and (15 in point_list): # Draw left leg
            draw_ellipse_and_line(draw, person_conf_multi, people_i, 11, 13, 15, point_color)
            part_count = part_count + 1
        if (12 in point_list) and (14 in point_list) and (16 in point_list): # Draw right leg
            draw_ellipse_and_line(draw, person_conf_multi, people_i, 12, 14, 16, point_color)
            part_count = part_count + 1
        if point_count >= point_min:
            people_real_num = people_real_num + 1
            for point_i in range(0, point_num):
                if person_conf_multi[people_i][point_i][0] + person_conf_multi[people_i][point_i][1] != 0: # If coordinates of point is (0, 0) == meaningless data
                    draw.ellipse(ellipse_set(person_conf_multi, people_i, point_i), fill=point_color)
                    people_x.append(person_conf_multi[people_i][point_i][0])
                    people_y.append(person_conf_multi[people_i][point_i][1])
            # Draw rectangle which include that people
            # draw.rectangle([min(people_x), min(people_y), max(people_x), max(people_y)], fill=point_color, outline=5)
            target_points.append((int(min(people_x)), int(min(people_y)), int(max(people_x)), int(max(people_y))))

        if part_count >= part_min:
            people_part_num = people_part_num + 1

    ### object-tracker ###
    if i == 0: # for frame 0. set tracker
        # Initial co-ordinates of the object to be tracked
        # Create the tracker object
        tracker = [dlib.correlation_tracker() for _ in range(len(target_points))]
        # Provide the tracker the initial position of the object
        [tracker[i].start_track(image, dlib.rectangle(*rect)) for i, rect in enumerate(target_points)]

    # Update the tracker
    for k in range(len(tracker)):
        tracker[k].update(image)
        # Get the position of th object, draw a
        # bounding box around it and display it.
        rect = tracker[k].get_position()
        pt1 = (int(rect.left()), int(rect.top()))
        pt2 = (int(rect.right()), int(rect.bottom()))
        draw.rectangle([rect.left(), rect.top(), rect.right(), rect.bottom()], fill='red', outline=5)
        # draw.rectangle([min(people_x), min(people_y), max(people_x), max(people_y)], fill='red', outline=5)
        # cv2.rectangle(image, pt1, pt2, (255, 255, 255), 3)
        print('Object ' + str(k) + ' tracked at [' + str(pt1) + ', ' + str(pt2) + ']')

    draw.text((0, 0), 'People(by point): ' + str(people_real_num) + ' (threshold = ' + str(point_min) + ')', (0,0,0), font=font)
    draw.text((0, 32), 'People(by line): ' + str(people_part_num) + ' (threshold = ' + str(part_min) + ')', (0,0,0), font=font)
    draw.text((0, 64), 'Frame: ' + str(i) + '/' + str(video_frame_number), (0,0,0), font=font)
    draw.text((0, 96), 'Total time required: ' + str(round(time.clock() - time_start, 1)) + 'sec', (0,0,0))

    print('people_real_num: ' + str(people_real_num))
    print('people_part_num: ' + str(people_part_num))
    print('frame: ' + str(i))

    image_img_numpy = np.asarray(image_img)

    pose_frame_list.append(image_img_numpy)

video_pose = ImageSequenceClip(pose_frame_list, fps=video.fps)
video_pose.write_videofile("testset/" + video_name + "_pose.mp4", fps=video.fps)

print("Time(s): " + str(time.clock() - time_start))
