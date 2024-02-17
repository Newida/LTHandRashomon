import pickle
import torch
import time
from pathlib import Path
import matplotlib.pyplot as plt
import itertools
import numpy as np
import torchvision
import torchvision.transforms as transforms
from resnet20 import Resnet_N_W
from Hparams import Hparams
from utils_Earlystopper import EarlyStopper
import utils
import routines
from captum.attr import IntegratedGradients, Saliency, DeepLift, NoiseTunnel
import torch.nn.functional as F


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
    train_seed=42,
    split_seed=0,
    rngCrop_seed=0,
    rngRandomHflip_seed=0,
    batch_size=2
)

#(down)load dataset cifar10
dataloaderhelper = utils.DataLoaderHelper(
    datasethparams=dataset_hparams
)
testset = dataloaderhelper.get_testset(data_path, dataloaderhelper.get_transform(False))
testloader = dataloaderhelper.get_test_loader(testset)

dissimilarity = 0
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

pairwise_euclid_dist = torch.nn.PairwiseDistance(p=2.0)
attribution_method = "grad"
noise_tunnel = False
models, all_model_stats, _1, _2, _3, _4 = routines.load_experiment("e7_1")
model1 = models[2]
model2 = models[3]
model1.eval()
model2.eval()
#print("Is model pruned?", Resnet_N_W.check_if_pruned(model1))
#print("Is model pruned?", Resnet_N_W.check_if_pruned(model2))
model1.remove_pruning()
model2.remove_pruning()
#print("Is model pruned?", Resnet_N_W.check_if_pruned(model1))
#print("Is model pruned?", Resnet_N_W.check_if_pruned(model2))


if attribution_method == "grad":
    attr_algo1 = Saliency(model1)
    attr_algo2 = Saliency(model2)
    kwargs = {"abs": False}
    if noise_tunnel == True:
        attr_algo1 = NoiseTunnel(attr_algo1)
        attr_algo2 = NoiseTunnel(attr_algo2)

elif attribution_method == "ig":
    attr_algo1 = IntegratedGradients(model1)
    attr_algo2 = IntegratedGradients(model2)
    #can specify baseline here but default is alread 0 so not necessary
    kwargs = {"n_steps": 100, "return_convergence_delta": True}
    if noise_tunnel == True:
        print("Cannot calculate smoothgrad for ig, since it takes to much memory")
        print("Ig without smoothgrad is performed.")

else:
    print("Attribution method not known. Choose either ig or grad.")

d = 0
normalization = 0
for data in testloader:
    inputs, labels = data
    inputs = inputs.to(device)
    inputs.requires_grad = True
    labels = labels.to(device)
    #calculate attribution for first model
    outputs1 = model1(inputs)
    labels1 = F.softmax(outputs1, dim=1)
    prediction_score, pred_labels_idx1 = torch.topk(labels1, 1)
    pred_labels_idx1.squeeze_()
    attributions1 = attr_algo1.attribute(inputs, target=pred_labels_idx1, **kwargs)

    outputs2 = model2(inputs)
    labels2 = F.softmax(outputs2, dim=1)
    prediction_score, pred_labels_idx2 = torch.topk(labels2, 1)
    pred_labels_idx2.squeeze_()
    attributions2 = attr_algo2.attribute(inputs, target=pred_labels_idx2, **kwargs)
    
    attributions1[attributions1 < 0] = 0
    attributions2[attributions2 < 0] = 0

    prediction_diff = pred_labels_idx1 - pred_labels_idx2
    attributions1[prediction_diff != 0] = 0
    attributions2[prediction_diff != 0] = 0

    distances = pairwise_euclid_dist(
        attributions1.flatten(start_dim=1, end_dim=-1),
        attributions2.flatten(start_dim=1, end_dim=-1)
    )
    d += torch.sum(distances)
    a = torch.sum(prediction_diff != 0)
    if a != 0:
        normalization += 2 - a.item()
        break
    normalization += 2

d /= normalization