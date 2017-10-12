# pose-tensorflow-video

Based on [pose-tensorflow](https://github.com/eldar/pose-tensorflow)

## Usage

Install [Docker](https://docker.com) and [Kitematic](https://kitematic.com/)

#### Pull docker image
```
$ docker pull jgravity/tf-opencv-jupyter:pose-video
$ docker run jgravity/tf-opencv-jupyter:pose-video
```

#### Download/Install code
```
# git clone https://github.com/PJunhyuk/pose-tensorflow-video

// If compile.sh permission ERROR
# chmod u+x compile.sh
# ./compile.sh

# cd models/coco
// If download_models.sh permission ERROR
# chmod u+x download_models.sh
# ./download_models.sh
# cd -
```
> If ```# ./download_models.sh ``` not works, use  ```# chmod u+x download_models.sh ``` ``` # ./download_models_wget.sh ```

###### Shorter version
```
# chmod u+x ./compile.sh && ./compile.sh && cd models/coco && chmod u+x download_models.sh && ./download_models.sh && cd -
```

#### Download videos in testset
```
# chmod u+x ./testset/download_testset_wget.sh && ./testset/download_testset_wget.sh
```

#### Multiperson pose detection in image
```
# TF_CUDNN_USE_AUTOTUNE=0 python3 demo/demo_multiperson.py {image_file_name}
```
> ex. testset/test_multi_00.png -> test_multi_00

#### Convert video frames to images
```
# python -c 'from video_pose import *; video2frame("{video_file_name}")'
```
> ex. testset/test_video_01.mov -> test_video_01
```
# python -c 'import time; start_time = time.clock(); from video_pose import *; video2frame("{video_file_name}"); print("Time(s): " + str(time.clock() - start_time))'
```
> With stopwatch

#### Convert video frames to images with pose
```
# python -c 'from video_pose import *; video2poseframe("{video_file_name}")'
```
> ex. testset/test_video_01.mov -> test_video_01
```
# python -c 'import time; start_time = time.clock(); from video_pose import *; video2poseframe("{video_file_name}"); print("Time(s): " + str(time.clock() - start_time))'
```
> With stopwatch

#### Convert video to video with pose
```
# python -c 'from video_pose import *; video2posevideo("{video_file_name}")'
```
> ex. testset/test_video_01.mov -> test_video_01

#### Tracking people
```
# python video_tracking.py -v '{video_file_name}'
```
> ``` # python video_tracking.py -v 'test_video_01' ```

## Environments

Use Docker [jgravity/tf-opencv-jupyter](https://hub.docker.com/r/jgravity/tf-opencv-jupyter/),

or install

- python 3.5.3
- opencv 3.1.0
- jupyter 4.2.1
- git 2.1.4
- tensorflow 1.3.0
- pip packages
  - scipy 0.19.1
  - scikit-image 0.13.1
  - matplotlib 2.0.2
  - pyYAML 3.12
  - easydict 1.7
  - Cython 0.27.1
  - munkres 1.0.12
  - moviepy 0.2.3.2
  - dlib 19.7.0
  - imageio 2.1.2

## Reference

### Test dataset
- testset/test_video_01: [Pedestrian overpass - original video (sample) - BriefCam Syndex](https://www.youtube.com/watch?v=aUdKzb4LGJI)
- testset/test_video_02: [Pedestrian Walking and Traffic Exit,Human Activity Recognition Video ,DataSet By UET Peshawar](https://www.youtube.com/watch?v=eZRLm7KK8HA)
### Citation
    @inproceedings{insafutdinov2017cvpr,
	    title = {ArtTrack: Articulated Multi-person Tracking in the Wild},
	    booktitle = {CVPR'17},
	    url = {http://arxiv.org/abs/1612.01465},
	    author = {Eldar Insafutdinov and Mykhaylo Andriluka and Leonid Pishchulin and Siyu Tang and Evgeny Levinkov and Bjoern Andres and Bernt Schiele}
    }

    @article{insafutdinov2016eccv,
        title = {DeeperCut: A Deeper, Stronger, and Faster Multi-Person Pose Estimation Model},
	    booktitle = {ECCV'16},
        url = {http://arxiv.org/abs/1605.03170},
        author = {Eldar Insafutdinov and Leonid Pishchulin and Bjoern Andres and Mykhaylo Andriluka and Bernt Schiele}
    }

### Code
[pose-tensorflow](https://github.com/eldar/pose-tensorflow) - Human Pose estimation with TensorFlow framework  
[object-tracker](https://github.com/bikz05/object-tracker) - Object Tracker written in Python using dlib and OpenCV
