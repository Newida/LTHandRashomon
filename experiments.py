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
    #initialize network
    #1. Setup hyperparameters
    training_hparams = Hparams.TrainingHparams(
        split_seed=dataloaderhelper.split_seed,
        data_order_seed=dataloaderhelper.data_order_seed,
        patience = 10,
        min_delta = 4,
        num_epoch = 200,
        gamma = 0.01,
        milestone_steps = [100, 150])
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
                             True)
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
    plt.yscale('log')
    plt.savefig(saving_experiments_path / "running_loss.png")
    plt.clf()
    plt.plot(x_iter, y_val_loss)
    plt.yscale('log')
    plt.savefig(saving_experiments_path / "vall_loss.png")
    return

"""
start = time.time()
stats = e1_train_val_loss("e1_4", "200 epoch training of network, version 3 of experiment 1 with different LR milestones and gamma set to 0.01 and [100, 150]")
end = time.time()
print("Time of Experiment 1:", end - start)
models, all_stats, _1, _2, _3, _4 = routines.load_experiment("e1")
model = models[0]
model.to(device)
print("Test_acc: ", routines.get_accuracy(device, model, testloader))
print("Train_acc: ",routines.get_accuracy(device, model, trainloader))
"""

def e2_rewind_iteration(name, description, rewind_iter):
    #initialize network
    #1. Setup hyperparameters
    training_hparams = Hparams.TrainingHparams(
        split_seed=dataloaderhelper.split_seed,
        data_order_seed=dataloaderhelper.data_order_seed,
        patience = 10,
        min_delta = 4,
        num_epoch = 2, #200
        gamma = 0.01,
        milestone_steps = [100, 150])
    pruning_hparams = Hparams.PruningHparams(
        pruning_stopper_patience = 3,
        pruning_stopper_min_delta = 4,
        max_pruning_level = 2, #15
        rewind_iter = rewind_iter,
        pruning_ratio = 0.2
    )
    model_structure, initializer, outputs = Resnet_N_W.get_model_from_name("resnet-20")
    model_hparams = Hparams.ModelHparams(
        model_structure, initializer, outputs, initialization_seed=0
        )
    #2. Setup model
    model = Resnet_N_W(model_hparams)
    #3. Train model
    early_stopper = EarlyStopper(
        model_hparams,
        patience=training_hparams.early_stopper_patience,
        min_delta=training_hparams.early_stopper_min_delta
        )
    pruning_stopper = EarlyStopper(
        model_hparams,
        patience=pruning_hparams.pruning_stopper_patience,
        min_delta=pruning_hparams.pruning_stopper_min_delta
        )
    models, all_model_stats, best_model = routines.imp(
        device,
        model,
        early_stopper, pruning_stopper,
        training_hparams, pruning_hparams,
        dataloaderhelper
        )
    #4. Save model and statistics
    routines.save_experiment(name,
                             description,
                             dataset_hparams,
                             training_hparams,
                             pruning_hparams,
                             model_hparams,
                             models,
                             all_model_stats,
                             False)
    #5. Plot some results
    L_pruning_level = []
    y_test_loss = []
    for L, stats in enumerate(all_model_stats):
        L_pruning_level.append(L)
        y_test_loss.append(stats[-1][1]['test_loss'])

    workdir = Path.cwd()
    experiments_path = workdir / "experiments"
    if not experiments_path.exists():
        raise ValueError("No exerpiment exists.")

    saving_experiments_path = experiments_path / name
    if not saving_experiments_path.exists():
        raise ValueError("Exerpiment does not exists.")
    
    plt.plot(L_pruning_level[1:], y_test_loss[1:])
    #test value of untrained network makes graph harder to see
    plt.savefig(saving_experiments_path / "test_loss.png")
    return 


start = time.time()
e2_rewind_iteration("e2_1", "rewind_iter = 0, pruing_ratio = 0.2", 0)
end = time.time()
print("Time of Experiment 2:", end - start)
models, all_stats, _1, _2, _3, _4 = routines.load_experiment("e2_1")
print("#models: ", len(models))
for L, model in enumerate(models):
    model.to(device)
    print("Pruning depth: ", L)
    print("Test_acc: ", routines.get_accuracy(device, model, testloader))
    print("Train_acc: ", routines.get_accuracy(device, model, trainloader))
    print("Density: ", Resnet_N_W.calculate_density(model))

def test_linear_mode_connectivity(name):
    workdir = Path.cwd()
    
    experiments_path = workdir / "experiments"
    if not experiments_path.exists():
        raise ValueError("No exerpiment exists.")

    saving_experiments_path = experiments_path / name
    if not saving_experiments_path.exists():
        raise ValueError("Exerpiment does not exists.")

    models, all_stats, _1, _2, _3, _4 = routines.load_experiment(saving_experiments_path)
    all_errors = []
    #since itertools pairwise is not available
    all_errors = list()
    a, b = itertools.tee(models[1:])
    next(b, None)
    for L, (model1, model2) in enumerate(zip(a, b)):
        print("Testing pruning depth: " + str(L))
        errors = routines.linear_mode_connected(
            device,
            model1, model2,
            dataloaderhelper)
        if L == 0:
            all_errors += errors
        else:
            print("Check errors: Expected close to 0:", all_errors[-1] - errors[0])
            all_errors += errors[1:]
        print("Got errors of: ", errors)

    length = len(models) - 2
    print("x:", np.linspace(0, length, num=11*length-1))
    print("y:", all_errors)
    plt.plot(np.linspace(0, length, num=11*length-1), all_errors) #11 since len(beta) = 11
    plt.savefig(saving_experiments_path / "linear_mode_connectivity.png")
    
#test_linear_mode_connectivity("e2_1")