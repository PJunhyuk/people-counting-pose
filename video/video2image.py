import numpy as np
import cv2

## Read video from file
video_input = cv2.VideoCapture('./testset/test_video_01.mov')

## Check information of video_input
print('video_input:', video_input)
print('type(video_input):', type(video_input)))

# Release everything if job is finished
video_input.release()
cv2.destroyAllWindows()
