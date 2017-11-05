from moviepy.editor import *

def read_video(video_name):
    # Read video from file
    video_name_input = 'testset/' + video_name
    video = VideoFileClip(video_name_input)
    return video

def resize_video(video, video_width):
    # Re-set size of video
    video = video.resize(width = video_width)
    print("Changed video size: [" + str(video.size[0]) + ", " + str(video.size[1]) + "]")
    return video

def background_subtraction(video):

if __name__ == '__main__':
    
