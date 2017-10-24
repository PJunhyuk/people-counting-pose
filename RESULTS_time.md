## RESULTS

> Print CPU info: ```# cat /proc/cpuinfo | grep 'model name' | uniq```

### video_tracking.py

#### DT
> CPU: Intel(R) Core(TM) i5-6600 CPU @ 3.30GHz
> GPU: NVIDIA GeForce GTX 750 Ti  

- test_video_01.mov
> 2.0MB, 72 frame, [1280, 720]

  - 1730.4sec - 1230a83eb89445f0f470553a44d582f825248e6f - test_video_01_tracking.mp4


- test_video_01f.mov
> 1.70MB, 733 frame, [640, 360]

  - 4152.6sec - 616f1d2e47ac792b89dd43e0a1441e8267c934f0 - test_video_01f_tracking_001.mp4
  - 1286.7sec - 3c68751f5181a0a01250cd28ccb4cdd9e9355227 - test_video_01f_tracking_005.mp4
    > ```# python video_tracking.py -f 'test_video_01f.mov' -w 360```

  - 2298.2sec - afcaf3c02ec18f3ff82e28769ca7287af1c3eb68

- test_video_03_01.mp4
> 2.3MB, 119 frame, [1920, 1080]

  - 6154.5sec - 68db3a271f17843917f8a614e66ebcbbfab7594a - test_video_03_01_tracking.mp4
  - 6383.5sec - 1e5425a3f8938c9b0e130d575f21bc8221deff8f - test_video_03_01_tracking_002.mp4
  - 747.9sec - fa7ce9a0c8bf5774ec891bc5d707f854d728f230 - test_video_03_01_tracking_003.mp4


- test_video_03f.mp4
> 64.3MB, 3639 frame, [1920, 1080]

  - 8556.9sec - 3c68751f5181a0a01250cd28ccb4cdd9e9355227 - test_video_03f_tracking_004.mp4
    > ```# python video_tracking.py -f 'test_video_03f.mp4' -w 360```


#### MCML
> CPU: Intel(r) Xeon(R) CPU E5-2687W v3 @ 3.10GHz

- test_video_01.mov

  - 140.0sec - 23b0a408ab14066e73b99fe5c272849b6bffe32e - test_video_01_tracking_006.mp4

- test_video_01f.mov

  - 435.0sec - 23b0a408ab14066e73b99fe5c272849b6bffe32e - test_video_01_tracking_007.mp4

- test_video_03f.mp4

  - 17059.9sec
