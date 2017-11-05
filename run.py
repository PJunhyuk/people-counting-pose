import argparse

import preprocessing

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--videoFile", help="Path to Video File")

    ## Preprocessing

    ### Read video from file
    if args["videoFile"] is not None:
        video_name = args["videoFile"]
    else:
        print("You have to input videoFile name")
        sys.exit(1)
    video = preprocessing.read_video(video_name)

    ### Print informations of video
    print("Input video size: [" + str(video.size[0]) + ", " + str(video.size[1]) + "]")

    ### Resize video
    if args["videoWidth"] is not None:
        video_width = int(args["videoWidth"])
        video = preprocessing.resize_video(video, video_width)

    ### background subtraction

    ## Human detection

    
