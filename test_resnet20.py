import torch
import numpy as np
from resnet20 import Resnet_N_W
from pathlib import Path
import torchvision
import torchvision.transforms as transforms
from Hparams import Hparams
from utils_DataLoader import DataLoaderHelper

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

#Test reintialization
plan, initializer, outputs = Resnet_N_W.get_model_from_name("resnet-20")
resnet20model = Resnet_N_W(plan, initializer, outputs)
resnet20model_copy = Resnet_N_W(plan, initializer, outputs)
resnet20model_copy.load_state_dict(resnet20model.state_dict())

#setting the path to store/load dataset cifar10
workdir = Path.cwd()
data_path = workdir / "datasets" / "cifar10"
if not data_path.exists():
    data_path.mkdir(parents=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
transform = transforms.Compose(
    [transforms.RandomCrop(32, padding=4),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize(mean, std)
     ])
dataset_hparams = Hparams.DatasetHparams()
dataloaderhelper = DataLoaderHelper(0, 0, dataset_hparams)
trainset = dataloaderhelper.get_trainset(data_path, transform)
trainset, valset = dataloaderhelper.split_train_val(trainset)
trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=dataset_hparams.batch_size,
                                         shuffle=True, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model_hparams = Hparams.ModelHparams()
training_hparams = Hparams.TrainingHparams(num_epoch=1, milestone_steps=[])
print("Before training:")
module_list = [resnet20model.conv, resnet20model.bn, resnet20model.blocks, resnet20model.fc]
with torch.no_grad():
    for module in module_list:
        if isinstance(module, torch.nn.Conv2d):
            print("CONV")
        elif isinstance(module, torch.nn.BatchNorm2d):
            print("BATCHNORM")
        elif isinstance(module, torch.nn.Sequential):
            print("BLOCKS:")
            for name, module_in_block in module.named_children():
                print(module_in_block)
        elif isinstance(module, torch.nn.Linear):
            print("LINEAR")
        else:
            print("UNEXPECTED")
            print(module)

"""
#train a single epoch
resnet20model.to(device)
resnet20model.train()
optimizer = Hparams.get_optimizer(resnet20model, training_hparams)
loss_criterion = Hparams.get_loss_criterion(training_hparams)
print("Started training ...")
running_loss = 0.0
for i, data in enumerate(trainloader):
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)    
    optimizer.zero_grad()
    outputs = resnet20model(inputs)
    loss = loss_criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    if i % 100 == 0:
        print(f'[{1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
        running_loss = 0.0
        
print("After training:")
with torch.no_grad():
    for name, module in resnet20model.named_modules():
        print("name:", name)
        print("module:", module)

resnet20model.reinitialize_model(resnet20model_copy)
print("After Reset")
with torch.no_grad():
    for name, module in resnet20model.named_modules():
        print("name:", name)
        print("module:", module)
"""