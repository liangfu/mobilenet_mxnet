#!/usr/bin/env python

"""
Conversion from pretrained mobilenet model using Gluon into mxnet defined symbols.
"""

import mxnet as mx

def rename_layer(layer_name):
    layer_id = 0
    to_newid = {"0":"1",
                "1":"2_1_dw","2":"2_1_sep","3":"2_2_dw","4":"2_2_sep",
                "5":"3_1_dw","6":"3_1_sep","7":"3_2_dw","8":"3_2_sep",
                "9":"4_1_dw","10":"4_1_sep","11":"4_2_dw","12":"4_2_sep",
                "13":"5_1_dw","14":"5_1_sep","15":"5_2_dw","16":"5_2_sep","17":"5_3_dw","18":"5_3_sep","19":"5_4_dw","20":"5_4_sep","21":"5_5_dw","22":"5_5_sep","23":"5_6_dw","24":"5_6_sep",
                "25":"6_1_dw","26":"6_1_sep",}
    if layer_name.startswith("batchnorm"):
        layer_id = layer_name.split("_")[0][9:]
        layer_name = layer_name.replace("batchnorm"+layer_id, "conv"+to_newid[layer_id]+"_bn")
        if layer_name.endswith("running_var") or layer_name.endswith("running_mean"):
            layer_name = layer_name.replace("running","moving")
    elif layer_name.startswith("conv"):
        layer_id = layer_name.split("_")[0][4:]
        layer_name = layer_name.replace("conv"+layer_id, "conv"+to_newid[layer_id])
    elif layer_name.startswith("dense"):
        layer_id = layer_name.split("_")[0][5:]
        layer_name = layer_name.replace("dense0", "fc7")
    if "_moving_" in layer_name:
        layer_name = "aux:"+layer_name
    else:
        layer_name = "arg:"+layer_name
    return layer_name

def main():
    save_dict = mx.nd.load("mobilenet-050-0000.params")
    keys = [rename_layer(key) for key in save_dict.keys()]
    values = save_dict.values()
    save_dict2 = dict(zip(keys, values))
    mx.nd.save("mobilenetv2-50-0000.params", save_dict2)

if __name__=="__main__":
    main()

