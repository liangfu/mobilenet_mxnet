#!/usr/bin/env bash

### convert tensorflow checkpoint files to IR
python MMdnn/mmdnn/conversion/_script/convertToIR.py -f tensorflow -d IR/mobilenet_v1_0.25_224 -n MobileNet/mobilenet_v1_0.25_224.ckpt.meta -w MobileNet/mobilenet_v1_0.25_224.ckpt --dstNodeName MobilenetV1/Predictions/Softmax

### convert IR to mxnet checkpoint files
python MMdnn/mmdnn/conversion/_script/IRToCode.py -f mxnet --IRModelPath IR/mobilenet_v1_0.25_224.pb --dstModelPath mobilenet_v1_025_224.py --IRWeightPath IR/mobilenet_v1_0.25_224.npy -dw IR/mobilenet_v1_0.25_224-0000.params

### convert IR to mxnet checkpoint files
python MMdnn/mmdnn/conversion/examples/mxnet/imagenet_test.py -n mobilenet_v1_025_224 -w IR/mobilenet_v1_0.25_224-0000.params --dump IR/mobilenet_v1_0.25_224

