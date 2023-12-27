from pathlib import Path
import torch
import torchvision
import torchvision.transforms as transforms
from resnet20 import Resnet_N_W
from Hparams import Hparams
from utils import EarlyStopper
import utils
import pickle
import numpy as np

try:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except NameError or ModuleNotFoundError:
    pass

#do training 
# TODO: be wary for the randomness in the training
# as it is important to identify different winning tickets later
# identify randomness: dataorder, TODO: find more

def train(device, model, rewind_iter, dataloaderhelper, training_hparams,
          early_stopper,
            calc_stats=True):
    model.to(device)
    model.train()
    optimizer = Hparams.get_optimizer(model, training_hparams)
    lr_scheduler = Hparams.get_lr_scheduler(optimizer, training_hparams)
    loss_criterion = Hparams.get_loss_criterion(training_hparams)
    trainloader = dataloaderhelper.trainloader
    valloader = dataloaderhelper.valloader
    testloader = dataloaderhelper.testloader
    #reset generator of trainloader to achive same data_order during training
    dataloaderhelper.reset_trainloader_generator(trainloader)
    
    all_stats = []
    iter = 0
    max_iter = dataloaderhelper.epochs_to_iter(training_hparams.num_epoch)
    if rewind_iter > max_iter:
        raise ValueError("rewind_iter must be smaller than " + str(max_iter))
    print("Started training for " + str(max_iter) + " iterations ...")
    running_loss = 0.0
    while True:
        for data in trainloader:
            #create rewind point
            if iter == rewind_iter:
                model_hparams = Hparams.ModelHparams(model.plan, model.initializer, model.outputs, model.weight_seed)
                rewind_point = Resnet_N_W(model_hparams)
                rewind_point.load_state_dict(model.state_dict())
            
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if iter % 100 == 0:
                print(f'[{iter}] loss: {running_loss:.3f}')
                if calc_stats:
                    stats = calculate_stats(device, model, loss_criterion,
                        valloader,
                        testloader,
                        running_loss
                        )
                    all_stats.append([iter, stats])
                    #check early_stopping
                    val_loss = stats[2]
                    print('[' + str(iter) + '] train_acc: ' + str(stats[1])
                          + 'val_loss: ' + str(stats[2]) + 'test_acc: ' + str(stats[5]))
                else:
                    val_loss = get_loss(device, model, valloader, loss_criterion)
                    all_stats.append([iter, {"running_loss": running_loss, "val_loss" : val_loss}])
                    print('[' + str(iter) + '] val_loss: ' + str(val_loss))

                if early_stopper.early_stop_val_loss(val_loss):
                    print("Stopped early")
                    print("Trained for " + str(iter) + " Iterations.")
                    return rewind_point, all_stats
            
            running_loss = 0.0
            if iter % dataloaderhelper.epochs_to_iter(1) == 0:
                #do a lr_step after every epoch
                lr_scheduler.step()
            iter += 1
            if iter >= max_iter:
                print("Trained for " + str(iter) + " Iterations.")
                return rewind_point, all_stats


def imp(model, random_state,
        training_hparams, pruning_hparams, saving_models_path,
        dataloaderhelper):
    #TODO: replace pruning level by early stopping
    #TODO: Add calculation of statistics
    max_pruning_level = pruning_hparams.max_pruning_level
    rewind_iter = pruning_hparams.rewind_iter
    with utils.TorchRandomSeed(random_state):
        #save initial model
        torch.save(model.state_dict(), saving_models_path / "resnet-0.pth")
        rewind_point, all_stats = train(
                    model,
                    rewind_iter,
                    training_hparams,
                    dataloaderhelper,
                    calc_stats = False
                )
        for L in range(1, max_pruning_level):
            #do training
            train(
                model,
                rewind_iter,
                training_hparams,
                dataloaderhelper
            )
            #pruning
            model.prune(
                prune_ratio = pruning_hparams.pruning_ratio,
                method = pruning_hparams.pruning_method
            )
            torch.save(model.state_dict(), saving_models_path / ("resnet-" + str(L + 1) + ".pth"))
            #rewind
            model.rewind(rewind_point)

def get_loss(device, model, dataloader, loss_criterion):
    with torch.no_grad():
        cumulated_loss = 0
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = loss_criterion(outputs, labels)
            cumulated_loss += loss.item()
            
    return cumulated_loss

def get_accuracy(device, model, dataloader):
    with torch.no_grad():
        correct = 0
        total = 0
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct/total

def get_loss_and_accuracy(device, model, dataloader, loss_criterion):
    with torch.no_grad():
        correct = 0
        total = 0
        cumulated_loss = 0
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            
            loss = loss_criterion(outputs, labels)
            cumulated_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return cumulated_loss, correct/total

def calculate_stats(device, model, loss_criterion,
                    valloader,
                     testloader,
                     running_loss
                     ):
    model.eval()
    #validation statistics
    val_loss, val_accuracy = get_loss_and_accuracy(device, model, valloader, loss_criterion)
    #test statistics
    test_loss, test_accuracy = get_loss_and_accuracy(device, model, testloader, loss_criterion)
    model.train()

    return {"running_loss": running_loss,
            "val_loss" : val_loss,
            "val_acc" : val_accuracy,
            "test_loss" : test_loss,
            "test_acc" : test_accuracy}

"""
models_path = workdir / "models"
if not models_path.exists():
    models_path.mkdir(parents=True)

saving_models_path = models_path / "experiment1"
if not saving_models_path.exists():
    saving_models_path.mkdir(parents=True)

#initialize hyperparemeters
training_hparams = Hparams.TrainingHparams(num_epoch=1, milestone_steps=[2])
pruning_hparams = Hparams.PruningHparams()
model_hparams = Hparams.ModelHparams()

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#create model
plan, initializer, outputs = Resnet_N_W.get_model_from_name("resnet-20")
resnet20model = Resnet_N_W(plan, initializer, model_hparams.initialization_seed, outputs)
#naming convention: resnet-N-W_<num_epoch>_<1.milestone>_<2.milestone>

import time
start = time.time()
imp(
    resnet20model,
    training_hparams,
    pruning_hparams,
    saving_models_path,
    trainloader,
    valloader,
    testloader,
    max_pruning_level = 1,
    rewind_iter = 20
    )
end = time.time()
print("Time of IMP:", end - start)
"""

def save_experiment(
        path,
        description,
        dataset_hparams, training_hparams, pruning_hparams, model_hparams,
        models, all_model_stats,
        override = False
):
    workdir = Path.cwd()
    experiments_path = workdir / "experiments"
    if not experiments_path.exists():
        experiments_path.mkdir(parents=True)

    saving_experiments_path = experiments_path / path
    if not saving_experiments_path.exists():
        saving_experiments_path.mkdir(parents=True)
    else:
        if not override:
            raise ValueError("Experiment would override other experiment. Cannot be saved.")

    path = saving_experiments_path

    with open(path / "ModelHparams.obj","wb") as f1:
        pickle.dump(model_hparams, f1)
    #L is the depth of the IMP iteration
    length = int(np.log10(len(models))) + 1
    for L, model in enumerate(models):
        model_number = str(L)
        model_name = "model" + "0" * (length-len(model_number)) + model_number + ".pth"
        torch.save(model.state_dict(), path / model_name)
    for L, stats in enumerate(all_model_stats):
        with open(path / ("stats" + str(L) + ".list"),"wb") as f2:
            pickle.dump(stats, f2)
    with open(path / "TrainingHparams.obj","wb") as f3:
        pickle.dump(training_hparams, f3)
    with open(path / "PruningHparams.obj","wb") as f4:
        pickle.dump(pruning_hparams, f4)
    with open(path / "DatasetHparams.obj","wb") as f5:
        pickle.dump(dataset_hparams, f5)
    with open(path / "Description.txt", "w") as f6:
        f6.write(description)

def load_experiment(path):
    workdir = Path.cwd()
    experiments_path = workdir / "experiments"
    path = experiments_path / path

    if not path.exists():
        raise ValueError("Experiment could not be found.")
    with open(path / "ModelHparams.obj",'rb') as f1:
        model_hparams = pickle.load(f1)
    models = []
    all_model_stats = []
    for p in sorted(path.iterdir()):
        if p.match('*.pth'):
            model = Resnet_N_W(model_hparams)
            try:
                #try loading unpruned model
                model.load_state_dict(torch.load(p))
            except:
                #loading pruned
                model.prune(1, "identity")
                model.load_state_dict(torch.load(p))
            models.append(model)
        if p.match('*.list'):
            with open(p,"rb") as f2:
                stats = pickle.load(f2)
                all_model_stats.append(stats)
    with open(path / "TrainingHparams.obj",'rb') as f3:
        training_hparams = pickle.load(f3)
    with open(path / "PruningHparams.obj",'rb') as f4:
        pruning_hparams = pickle.load(f4)
    with open(path / "DatasetHparams.obj",'rb') as f5:
        dataset_hparams = pickle.load(f5)

    return models, all_model_stats, model_hparams, training_hparams, pruning_hparams, dataset_hparams