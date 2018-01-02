#!/usr/bin/env bash

### convert tensorflow checkpoint files to IR
# python -m mmdnn.conversion._script.convertToIR -f tensorflow -d IR/mobilenet_v1_0.25_224 -n MobileNet/mobilenet_v1_0.25_224.ckpt.meta -w MobileNet/mobilenet_v1_0.25_224.ckpt --dstNodeName MobilenetV1/Predictions/Softmax

### convert IR to mxnet checkpoint files
python -m mmdnn.conversion._script.IRToCode -f mxnet --IRModelPath IR/mobilenet_v1_0.25_224.pb --dstModelPath IR/mobilenet_v1_0.25_224.py --IRWeightPath IR/mobilenet_v1_0.25_224.npy -dw IR/mobilenet_v1_0.25_224-0000.params

### convert IR to mxnet checkpoint files
# python -m mmdnn.conversion.examples.mxnet.imagenet_test -n mxnet_

