#test dataloader helper
from pathlib import Path
import torch
import torchvision
import torchvision.transforms as transforms
from resnet20 import Resnet_N_W
from Hparams import Hparams
from utils_DataLoader import DataLoaderHelper, TrainLoader
from utils_Earlystopper import EarlyStopper
import numpy as np

#setting the path to store/load dataset cifar10
workdir = Path.cwd()
data_path = workdir / "datasets" / "cifar10"
if not data_path.exists():
    data_path.mkdir(parents=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#data augmentation
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
transform = transforms.Compose(
    [transforms.RandomCrop(32, padding=4),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize(mean, std)
     ])



dataset_hparams = Hparams.DatasetHparams()

#(down)load dataset cifar10
dataloaderhelper = DataLoaderHelper(0, 0, dataset_hparams)

trainset = dataloaderhelper.get_trainset(data_path, transform)
testset = dataloaderhelper.get_testset(data_path, transform)

trainset, valset = dataloaderhelper.split_train_val(trainset)

print("lent:", len(trainset))
print("lenv:", len(valset))

#Test Trainloader
traingenerator = torch.Generator()
traingenerator.manual_seed(0)
traingenerator2 = torch.Generator()
traingenerator2.manual_seed(0)
trainloader = TrainLoader(trainset, 128, traingenerator)
for images, labels in trainloader:
    print(images.shape)
    break

orig_tensor = torch.tensor([1,2,3,4,5])
torch.manual_seed(0)
shuffled_tensor = orig_tensor.clone()
indices = torch.randperm(shuffled_tensor.size((0)))
shuffled_tensor = shuffled_tensor[indices]
print("First 10 permuted indices:", shuffled_tensor)
shuffled_tensor2 = orig_tensor.clone()
indices2 = torch.randperm(shuffled_tensor.size((0)))
shuffled_tensor2 = shuffled_tensor[indices2]
print("First 10 permuted indices:", shuffled_tensor2)
#print("Should be 0:", torch.norm(indices1 - indices1_2))
#permutes the indices from 0 to len(trainset)-1