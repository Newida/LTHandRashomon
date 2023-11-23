#test dataloader helper
from pathlib import Path
import torch
import torchvision
import torchvision.transforms as transforms
from resnet20 import Resnet_N_W
from Hparams import Hparams
from utils_DataLoader import DataLoaderHelper
from utils_Earlystopper import EarlyStopper

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