import torch
from resnet20 import Resnet_N_W

#test modelnames:
#false cases
print("False cases:")
print(Resnet_N_W.is_valid_model_name("resnet"))
print(Resnet_N_W.is_valid_model_name("resnet-"))
print(Resnet_N_W.is_valid_model_name("resnet-18"))
print(Resnet_N_W.is_valid_model_name("resnet-1-20"))

#true cases
print("True cases:")
print(Resnet_N_W.is_valid_model_name("resnet-20"))
print(Resnet_N_W.is_valid_model_name("resnet-20-16"))
print(Resnet_N_W.is_valid_model_name("resnet-20-24"))