# Result_log

> Print CPU info: ```# cat /proc/cpuinfo | grep 'model name' | uniq```

### Case 1: Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz

#### Environments
- `Window10 HOME`
- `Docker` with `Oracle VM VirtualBox`

### Results

`# python video_tracking.py -f test_video_01.mov`
> Video size: [1280, 720] / Total frame: 72 / Total required time: 1310.6sec

`# python video_tracking.py -f test_video_01f.mov`
> Video size: [640, 360] / Total frame: 733 / Total required time: 3426.5sec
