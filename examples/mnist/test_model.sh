#!/usr/bin/env sh
# set -e

./build/tools/caffe.bin test --model=examples/mnist/lenet.prototxt --weights=examples/mnist/lenet_iter_5000.caffemodel --iterations 100
