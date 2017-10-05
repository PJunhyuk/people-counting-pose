import sys
import os

import math

import imageio
from moviepy.editor import *

import time

def read_video(video_name):
    # Read video from file
    video_name_input = 'testset/' + video_name + '.mov'
    video = VideoFileClip(video_name_input)
    return video

def video2frame(video_name):
    video = read_video(video_name)

    video_frame_number = int(video.duration * video.fps) ## duration: second / fps: frame per second
    video_frame_ciphers = math.ceil(math.log(video_frame_number, 10)) ## ex. 720 -> 3

    if not os.path.exists('testset/' + video_name):
        os.makedirs('testset/' + video_name)

    for i in range(0, video_frame_number):
        video.save_frame('testset/' + video_name + '/frame_' + str(i).zfill(video_frame_ciphers) + '.jpg', i/video.fps)

def video2poseframe(video_name):
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

    from PIL import Image, ImageDraw

    import random

    cfg = load_config("demo/pose_cfg_multi.yaml")

    dataset = create_dataset(cfg)

    sm = SpatialModel(cfg)
    sm.load()

    # Load and setup CNN part detector
    sess, inputs, outputs = predict.setup_pose_prediction(cfg)

    ################

    video = read_video(video_name)

    video_frame_number = int(video.duration * video.fps) ## duration: second / fps: frame per second
    video_frame_ciphers = math.ceil(math.log(video_frame_number, 10)) ## ex. 720 -> 3

    if not os.path.exists('testset/' + video_name):
        os.makedirs('testset/' + video_name)

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

        print('person_conf_multi: ')
        print(type(person_conf_multi))
        print(person_conf_multi)

        # Add library to save image
        image_img = Image.fromarray(image)

        # Save image with points of pose
        draw = ImageDraw.Draw(image_img)

        people_num = 0
        point_num = 17
        print('person_conf_multi.size: ')
        print(person_conf_multi.size)
        people_num = person_conf_multi.size / (point_num * 2)
        people_num = int(people_num)
        print('people_num: ')
        print(people_num)

        point_i = 0 # index of points
        point_r = 5 # radius of points

        people_real_num = 0
        for people_i in range(0, people_num):
            point_color_r = random.randrange(0, 256)
            point_color_g = random.randrange(0, 256)
            point_color_b = random.randrange(0, 256)
            point_color = (point_color_r, point_color_g, point_color_b, 255)
            point_count = 0
            for point_i in range(0, point_num):
                if person_conf_multi[people_i][point_i][0] + person_conf_multi[people_i][point_i][1] != 0: # If coordinates of point is (0, 0) == meaningless data
                    point_count = point_count + 1
            if point_count > 5: # If there are more than 5 point in person, we define he/she is REAL PERSON
                people_real_num = people_real_num + 1
                for point_i in range(0, point_num):
                    draw.ellipse((person_conf_multi[people_i][point_i][0] - point_r, person_conf_multi[people_i][point_i][1] - point_r, person_conf_multi[people_i][point_i][0] + point_r, person_conf_multi[people_i][point_i][1] + point_r), fill=point_color)

        print('people_real_num: ')
        print(people_real_num)

        video_name_result = 'testset/' + video_name + '/frame_pose_' + str(i).zfill(video_frame_ciphers) + '.jpg'
        image_img.save(video_name_result, "JPG")


def video2posevideo(video_name):
    time_start = time.clock()

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

    cfg = load_config("demo/pose_cfg_multi.yaml")

    dataset = create_dataset(cfg)

    sm = SpatialModel(cfg)
    sm.load()

    draw_multi = PersonDraw()

    # Load and setup CNN part detector
    sess, inputs, outputs = predict.setup_pose_prediction(cfg)

    ################

    video = read_video(video_name)

    video_frame_number = int(video.duration * video.fps) ## duration: second / fps: frame per second
    video_frame_ciphers = math.ceil(math.log(video_frame_number, 10)) ## ex. 720 -> 3

    if not os.path.exists('testset/' + video_name):
        os.makedirs('testset/' + video_name)

    pose_frame_list = []

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

        print('person_conf_multi: ')
        print(type(person_conf_multi))
        print(person_conf_multi)

        # Add library to save image
        image_img = Image.fromarray(image)

        # Save image with points of pose
        draw = ImageDraw.Draw(image_img)

        people_num = 0
        point_num = 17

        people_num = person_conf_multi.size / (point_num * 2)
        people_num = int(people_num)
        print('people_num: ' + str(people_num))

        point_i = 0 # index of points
        point_r = 3 # radius of points
        point_min = 10 # threshold of points - If there are more than point_min points in person, we define he/she is REAL PERSON

        people_real_num = 0
        for people_i in range(0, people_num):
            point_color_r = random.randrange(0, 256)
            point_color_g = random.randrange(0, 256)
            point_color_b = random.randrange(0, 256)
            point_color = (point_color_r, point_color_g, point_color_b, 255)
            point_count = 0
            for point_i in range(0, point_num):
                if person_conf_multi[people_i][point_i][0] + person_conf_multi[people_i][point_i][1] != 0: # If coordinates of point is (0, 0) == meaningless data
                    point_count = point_count + 1
            if point_count >= point_min:
                people_real_num = people_real_num + 1
                for point_i in range(0, point_num):
                    draw.ellipse((person_conf_multi[people_i][point_i][0] - point_r, person_conf_multi[people_i][point_i][1] - point_r, person_conf_multi[people_i][point_i][0] + point_r, person_conf_multi[people_i][point_i][1] + point_r), fill=point_color)

        draw.text((0, 0), 'People_real_num: ' + str(people_real_num), (0,0,0), font=font)
        draw.text((0, 32), 'Frame: ' + str(i) + '/' + str(video_frame_number), (0,0,0), font=font)
        draw.text((0, 64), 'Total time required: ' + str(round(time.clock() - time_start, 1)) + 'sec = ' + str(round((time.clock() - time_start) / 60), 1) + 'min', (0,0,0))

        print('people_real_num: ' + str(people_real_num))
        print('frame: ' + str(i))

        image_img_numpy = np.asarray(image_img)

        pose_frame_list.append(image_img_numpy)

    video_pose = ImageSequenceClip(pose_frame_list, fps=video.fps)
    video_pose.write_videofile("testset/" + video_name + "_pose.mp4", fps=video.fps)

    print("Time(s): " + str(time.clock() - time_start))
