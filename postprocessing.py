from moviepy.editor import *

def frame_to_video(frame_list, video_file_name):
    video_pose = ImageSequenceClip(frame_list, fps=video.fps)
    video_output_name = video_name.split('.')[0]
    video_pose.write_videofile("testset/" + video_output_name + "_tracking." + video_type, fps=video.fps, progress_bar=False)

def __init__(self, frame_list, video_file_name):
