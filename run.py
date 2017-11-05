import argparse

import time
time_start = time.time()

import preprocessing

if __name__ == '__main__':

    ## Argument parsing

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--videoFile", help="Path to Video File")
    parser.add_argument('-w', "--videoWidth", help="Width of Output Video")

    if args["videoFile"] is not None:
        video_file_name = args["videoFile"]
    else:
        print("You have to input videoFile name")
        sys.exit(1)

    if args["videoWidth"] is not None:
        video_width = int(args["videoWidth"])
    else:
        video_width = 0

    ## Preprocessing

    video, total_frame_number = preprocessing(video_file_name, video_width)

#    for i in range(0, total_frame_number):

        ## Human detection

    ## Poseprocessing

#    poseprocessing(video)

    print("Total time required(s): " + str(time.time() - time_start))
