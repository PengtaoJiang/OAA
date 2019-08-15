#!/bin/bash

set -x

LOG="logs/cls_`date +%Y-%m-%d_%H-%M-%S`.txt"
exec &> >(tee -a "$LOG")
export PYTHONPATH=voc_layer.py:$PYTHONPATH
../../build/tools/caffe train --solver $1/solver_integral.prototxt --weights VGG_ILSVRC_16_layers.caffemodel --gpu $2
