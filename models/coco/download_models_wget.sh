#!/bin/sh

wget "http://datasets.d2.mpi-inf.mpg.de/deepercut-models-tensorflow/coco-resnet-101.data-00000-of-00001"
wget "http://datasets.d2.mpi-inf.mpg.de/deepercut-models-tensorflow/coco-resnet-101.meta"
wget "http://datasets.d2.mpi-inf.mpg.de/deepercut-models-tensorflow/coco-resnet-101.index"

wget "http://datasets.d2.mpi-inf.mpg.de/deepercut-models-tensorflow/pairwise_coco.tar.gz"
tar xvzf pairwise_coco.tar.gz
