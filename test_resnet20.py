import torch
import numpy as np
from resnet20 import Resnet_N_W

#test is_valid_model_name:
#false cases
print("Testing is_valid_model_name:")
print("-False cases:")
print(Resnet_N_W.is_valid_model_name("resnet"))
print(Resnet_N_W.is_valid_model_name("resnet-"))
print(Resnet_N_W.is_valid_model_name("resnet-18"))
print(Resnet_N_W.is_valid_model_name("resnet-2-20"))

#true cases
print("-True cases:")
print(Resnet_N_W.is_valid_model_name("resnet-20"))
print(Resnet_N_W.is_valid_model_name("resnet-20-16"))
print(Resnet_N_W.is_valid_model_name("resnet-20-24"))

print("-"*20)
#test is_valid_initalizer:
#false cases
print("Testing is_valid_initalizer")
print("-False cases:")
print(Resnet_N_W.is_valid_initalizer("resnet"))

#true cases
print("-True cases:")
print(Resnet_N_W.is_valid_initalizer("kaiming_normal"))
print(Resnet_N_W.is_valid_initalizer("kaiming_uniform"))

print("-"*20)
#test get_model_from_name:
print("Testing get_model_from_name")
#regular cases
print("-Testing regular cases:")
print(Resnet_N_W.get_model_from_name("resnet-20"))
print(Resnet_N_W.get_model_from_name("resnet-20", "kaiming_normal"))
print(Resnet_N_W.get_model_from_name("resnet-20-16", "kaiming_uniform"))
print(Resnet_N_W.get_model_from_name("resnet-20-16", "kaiming_normal", outputs=100))
print(Resnet_N_W.get_model_from_name("resnet-32"))
print(Resnet_N_W.get_model_from_name("resnet-44"))
print(Resnet_N_W.get_model_from_name("resnet-56"))

#error cases
print("Testing error inputs:")
try:
    print(Resnet_N_W.get_model_from_name("resnet-18-10"))
except ValueError:
    print("ValueError raised")
try:
    print(Resnet_N_W.get_model_from_name("resnet-20", "somethingesle"))
except ValueError:
    print("ValueError raised")

print("-"*20)

#test Resnet:
print("Testing Resnet:")
print("-Testing structure")
plan, initializer, outputs = Resnet_N_W.get_model_from_name("resnet-20")
resnet20model = Resnet_N_W(plan, initializer, outputs)
print(resnet20model.blocks)
print("-Testing forwardstep")
input = torch.unsqueeze(torch.stack([torch.eye(32),torch.eye(32),torch.eye(32)]), 0)
resnet20model(input)
print("-"*20)