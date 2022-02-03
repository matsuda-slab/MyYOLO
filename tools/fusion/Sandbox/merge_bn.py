import os, sys
import torch
import numpy as np
import re 
from collections import OrderedDict

KERNEL_DW   = 3
KERNEL_PW   = 1
#===============================================================================
# パラメータのロード
#===============================================================================
weights_path = sys.argv[1]

SEP = True if 'sep' in weights_path else False

params = torch.load(weights_path, map_location='cpu')

#kernel_3_3_layer = [1,2,3,4,5,6,7,9,12]
#kernel_1_1_layer = [8,11]
#non_bn_layer = [10,13]
kernel_3_3_layer = [1]
kernel_1_1_layer = []
non_bn_layer = []

merge_weight_dict = OrderedDict()#merge後の辞書型

print("step 1/3 layer",kernel_3_3_layer)
for now_layer in kernel_3_3_layer:
    print(now_layer)
    ###depth wise###
    weight    = params["conv"+str(now_layer)+".conv_dw.weight"]
    bn_weight = params["conv"+str(now_layer)+".bn_dw.weight"]
    bn_bias   = params["conv"+str(now_layer)+".bn_dw.bias"]
    bn_mean   = params["conv"+str(now_layer)+".bn_dw.running_mean"]
    bn_var    = params["conv"+str(now_layer)+".bn_dw.running_var"]
    #重みを結合
    CHANNELS = weight.shape[0] #input chanel
    merge_weight = torch.zeros(CHANNELS, 1, KERNEL_DW, KERNEL_DW)
    merge_bias = torch.zeros(CHANNELS)                           
    for c in range(CHANNELS):
        merge_weight[c][0] = weight[c][0]         * bn_weight[c]   / torch.sqrt(bn_var[c]   + sys.float_info.min)        
        merge_bias[c] = bn_bias[c]    - (bn_weight[c]   * bn_mean[c])  / torch.sqrt(bn_var[c]   + sys.float_info.min)
    merge_weight_dict["conv"+str(now_layer)+".conv_dw.weight"] = merge_weight
    merge_weight_dict["conv"+str(now_layer)+".conv_dw.bias"] = merge_bias

    ###point wise###
    weight_pw      = params["conv"+str(now_layer)+".conv_pw.weight"]     
    bn_weight_pw   = params["conv"+str(now_layer)+".bn_pw.weight"]       
    bn_bias_pw     = params["conv"+str(now_layer)+".bn_pw.bias"]         
    bn_mean_pw     = params["conv"+str(now_layer)+".bn_pw.running_mean"] 
    bn_var_pw      = params["conv"+str(now_layer)+".bn_pw.running_var"]  
    #重みを結合
    OUTPUT_CHANNELS = weight_pw.shape[0] #output chanel
    merge_weight_pw = torch.zeros(OUTPUT_CHANNELS, CHANNELS, KERNEL_PW, KERNEL_PW)
    merge_bias_pw = torch.zeros(OUTPUT_CHANNELS)
    for oc in range(OUTPUT_CHANNELS):
        for ic in range(CHANNELS):
            merge_weight_pw[oc][ic][0][0] = weight_pw[oc][ic][0][0] * bn_weight_pw[oc] / torch.sqrt(bn_var_pw[oc] + sys.float_info.min)
        merge_bias_pw[oc] = bn_bias_pw[oc] - (bn_weight_pw[oc] * bn_mean_pw[oc]) / torch.sqrt(bn_var_pw[oc] + sys.float_info.min)
    merge_weight_dict["conv"+str(now_layer)+".conv_pw.weight"] = merge_weight_pw 
    merge_weight_dict["conv"+str(now_layer)+".conv_pw.bias"] = merge_bias_pw
    

print("step 2/3  layer",kernel_1_1_layer)
for now_layer in kernel_1_1_layer:
    print(now_layer)
    #point wise 流用
    weight_pw      = params["conv"+str(now_layer)+".conv.weight"]                                                                   
    bn_weight_pw   = params["conv"+str(now_layer)+".bn.weight"]                                                                     
    bn_bias_pw     = params["conv"+str(now_layer)+".bn.bias"]                                                                       
    bn_mean_pw     = params["conv"+str(now_layer)+".bn.running_mean"]                                                               
    bn_var_pw      = params["conv"+str(now_layer)+".bn.running_var"]                                                                
    #重みを結合
    CHANNELS =  weight_pw.shape[1]  #input chanel                                                                                             
    OUTPUT_CHANNELS = weight_pw.shape[0]  #output chanel                                                              
    merge_weight_pw = torch.zeros(OUTPUT_CHANNELS, CHANNELS, KERNEL_PW, KERNEL_PW)                                                     
    merge_bias_pw = torch.zeros(OUTPUT_CHANNELS)                                                                                       
    for oc in range(OUTPUT_CHANNELS):                                                                                                  
        for ic in range(CHANNELS):                                                                                                     
            merge_weight_pw[oc][ic][0][0] = weight_pw[oc][ic][0][0] * bn_weight_pw[oc] / torch.sqrt(bn_var_pw[oc] + sys.float_info.min)
        merge_bias_pw[oc] = bn_bias_pw[oc] - (bn_weight_pw[oc] * bn_mean_pw[oc]) / torch.sqrt(bn_var_pw[oc] + sys.float_info.min)
    merge_weight_dict["conv"+str(now_layer)+".conv.weight"] = merge_weight_pw 
    merge_weight_dict["conv"+str(now_layer)+".conv.bias"] = merge_bias_pw


print("step 3/3 layer",non_bn_layer)
for now_layer in non_bn_layer:
    print(now_layer)
    merge_weight_dict["conv"+str(now_layer)+".weight"] = params["conv"+str(now_layer)+".weight"]
    merge_weight_dict["conv"+str(now_layer)+".bias"] = params["conv"+str(now_layer)+".bias"]


torch.save(merge_weight_dict,"./weights/merge.pt")
