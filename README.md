<img src="/samples/Odin_squared.png" width="200">

# Odin
### Pose estimation-based tracking and counting of people in videos

## Demo
<img src="/samples/sample_results.gif" width="600">  

[Demo on YouTube](http://www.youtube.com/watch?v=5lSUhCjgD7g)  
[Paper Abstract](https://github.com/PJunhyuk/people-counting-pose/blob/master/samples/Pose%20estimation-based%20tracking%20and%20counting%20of%20people%20in%20videos.pdf)

## Usage

Uses [Docker](https://docker.com)  

#### Pull docker image
```
$ docker pull jgravity/tensorflow-opencv:odin
$ docker run -it --name odin jgravity/tensorflow-opencv:odin bin/bash
```

#### Download/Install code
```
# git clone https://github.com/PJunhyuk/people-counting-pose
# cd people-counting-pose
# chmod u+x ./compile.sh && ./compile.sh && cd models/coco && chmod u+x download_models_wget.sh && ./download_models_wget.sh && cd -
```

#### Download sample videos in testset
```
# cd testset && chmod u+x ./download_testset_wget.sh && ./download_testset_wget.sh && cd -
```

#### Tracking people
```
# python video_tracking.py -f '{video_file_name}'
```
> Qualified supporting video type: mov, mp4  
> You have to put target video file in ./testset folder  

###### Arguments
> -f, --videoFile = Path to Video File  
> -w, --videoWidth = Width of Output Video  
> -o, --videoType = Extension of Output Video

###### Example
```
# python video_tracking.py -f 'test_video_01f.mov'
```

#### Check results
```
> docker cp odin:/people-counting-pose/testset/{video_file_name} ./
```

#### Just get pose of people (without tracking)
```
# python video_pose.py -f '{video_file_name}'
```
> Qualified supporting video type: mov, mp4

## Dependencies

Use Docker [jgravity/tensorflow-opencv](https://hub.docker.com/r/jgravity/tensorflow-opencv/),

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

## Results (time required)

Check [results_log](https://github.com/PJunhyuk/people-counting-pose/blob/master/results_log.md)

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

## LICENCE
[Apache License 2.0](https://github.com/PJunhyuk/people-counting-pose/blob/master/LICENSE)
