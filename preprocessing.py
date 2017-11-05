from moviepy.editor import *


def read_video(video_file_name):
    # Read video from file
    video_file_route = 'testset/' + video_file_name
    video = VideoFileClip(video_file_route)
    total_frame_number = int(video.duration * video.fps)
    return video, total_frame_number


def print_video_info(video):
    print("--Original Video Information--")
    print("Size: [" + str(video.size[0]) + ", " + str(video.size[1]) + "]")
    print("Duration(s): " + str(video.duration))
    print("FPS: " + str(video.fps))
    print("Total frame number: " + str(video.duration * video.fps))


def resize_video(video, video_width):
    # Re-set size of video
    video = video.resize(width = video_width)
    print("Changed video size: [" + str(video.size[0]) + ", " + str(video.size[1]) + "]")
    return video


def background_subtraction(video):
    import cv2

    # TODO

    return video


def __init__(self, video_file_name, video_width):
    video, total_frame_number = read_video(video_file_name)
    print("read_video complete!")

    print_video_info(video)
    print("print_video_info complete!")

    if video_width != 0:
        resize_video(video, video_width)
        print("resize_video complete!")

    video = background_subtraction(video)
    print("background_subtraction complete!")

    return video, total_frame_number
