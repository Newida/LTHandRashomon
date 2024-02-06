import pickle
import time
import itertools
import numpy as np
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import captum
from captum.attr import visualization as viz
from captum.attr import IntegratedGradients, Saliency, DeepLift, NoiseTunnel


from resnet20 import Resnet_N_W
from Hparams import Hparams
from utils_Earlystopper import EarlyStopper
import utils
import routines

try:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except NameError or ModuleNotFoundError:
    pass
#setting the path to store/load dataset cifar10
workdir = Path.cwd()
data_path = workdir / "datasets" / "cifar10"
if not data_path.exists():
    data_path.mkdir(parents=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_hparams = Hparams.DatasetHparams(
    test_seed=0,
    val_seed=0,
    train_seed=0,
    split_seed=0,
    rngCrop_seed=0,
    rngRandomHflip_seed=0,
    batch_size=2
)
dataloaderhelper = utils.DataLoaderHelper(
    datasethparams=dataset_hparams
)
testset = dataloaderhelper.get_testset(data_path, dataloaderhelper.get_transform(False))
testloader = dataloaderhelper.get_test_loader(testset)

for data in testloader:
    batch, label = data
    break

print(batch.shape)
batch = torch.flatten(batch, start_dim=1, end_dim=-1)
print(batch.shape)
