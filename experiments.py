import pickle
import torch
from pathlib import Path
import torchvision
import torchvision.transforms as transforms
from resnet20 import Resnet_N_W
from Hparams import Hparams
from utils_Earlystopper import EarlyStopper
import utils
import routines
import matplotlib.pyplot as plt

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

random_state = 0 #TODO: add to commandline arguments later

with utils.TorchRandomSeed(random_state):
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
    dataloaderhelper = utils.DataLoaderHelper(
        split_seed=0,
        data_order_seed=0,
        val_seed=0,
        test_seed=0,
        datasethparams=dataset_hparams
        )
    trainset = dataloaderhelper.get_trainset(data_path, transform)
    testset = dataloaderhelper.get_testset(data_path, transform)
    trainset, valset = dataloaderhelper.split_train_val(trainset)
    trainloader = dataloaderhelper.get_train_loader(trainset)
    testloader = dataloaderhelper.get_test_loader(testset)
    valloader = dataloaderhelper.get_validation_loader(valset)

def e1_train_val_loss(name, description):
    #Look at training routine
    #initialize network
    #1. Setup hyperparameters
    training_hparams = Hparams.TrainingHparams(
        split_seed=dataloaderhelper.split_seed,
        data_order_seed=dataloaderhelper.data_order_seed,
        num_epoch=200)
    pruning_hparams = Hparams.PruningHparams()
    model_structure, initializer, outputs = Resnet_N_W.get_model_from_name("resnet-20")
    model_hparams = Hparams.ModelHparams(
        model_structure, initializer, outputs, initialization_seed=0)
    #2. Setup model
    model = Resnet_N_W(model_hparams)
    #3. Train model
    early_stopper = EarlyStopper(
        model_hparams,
        patience=training_hparams.early_stopper_patience,
        min_delta=training_hparams.early_stopper_min_delta)
    
    _, all_stats, best_model = routines.train(device,
        model,
        0,
        dataloaderhelper,
        training_hparams,
        early_stopper,
        True
        )
    #4. Save model and statistics
    routines.save_experiment(name,
                             description,
                             dataset_hparams,
                             training_hparams,
                             pruning_hparams,
                             model_hparams,
                             [best_model],
                             [all_stats],
                             False)
    #5. Plot some results
    x_iter = []
    y_running_loss = []
    y_val_loss = []
    for stats in all_stats:
        x_iter.append(stats[0])
        stats = stats[1]
        y_running_loss.append(stats['running_loss'])
        y_val_loss.append(stats['val_loss'])

    workdir = Path.cwd()
    experiments_path = workdir / "experiments"
    if not experiments_path.exists():
        raise ValueError("No exerpiment exists.")

    saving_experiments_path = experiments_path / name
    if not saving_experiments_path.exists():
        raise ValueError("Exerpiment does not exists.")
    
    plt.plot(x_iter, y_running_loss)
    plt.savefig(saving_experiments_path / "running_loss.png")
    plt.clf()
    plt.plot(x_iter, y_val_loss)
    plt.savefig(saving_experiments_path / "vall_loss.png")
    return all_stats

import time
start = time.time()
stats = e1_train_val_loss("e1", "200 epoch training of network")
end = time.time()
print("Time of Experiment 1:", end - start)
models, all_stats, _1, _2, _3, _4 = routines.load_experiment("e1")
model = models[0]
model.to(device)
print("Test_acc: ", routines.get_accuracy(device, model, testloader))
print("Train_acc: ",routines.get_accuracy(device, model, trainloader))