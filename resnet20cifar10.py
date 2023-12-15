from pathlib import Path
import torch
import torchvision
import torchvision.transforms as transforms
from resnet20 import Resnet_N_W
from Hparams import Hparams
from utils import EarlyStopper
import utils

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

random_state = 0
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
    dataloaderhelper = utils.DataLoaderHelper(0, 0, dataset_hparams)
    trainset = dataloaderhelper.get_trainset(data_path, transform)
    testset = dataloaderhelper.get_testset(data_path, transform)
    trainset, valset = dataloaderhelper.split_train_val(trainset)
    trainloader = dataloaderhelper.get_train_loader(trainset)
    testloader = dataloaderhelper.get_test_loader(testset)
    valloader = dataloaderhelper.get_validation_loader(valset)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#create model
plan, initializer, outputs = Resnet_N_W.get_model_from_name("resnet-20")
resnet20model = Resnet_N_W(plan, initializer, outputs)
#naming convention: resnet-N-W_<num_epoch>_<1.milestone>_<2.milestone>

#do training 
#TODO: be wary for the randomness in the training
# as it is important to identify different winning tickets later
# identify randomness: dataorder, TODO: find more

early_stopper = EarlyStopper(patience=1, min_delta=0)
def train(model, training_hparams, trainloader, valloader):
    model.to(device)
    model.train()
    optimizer = Hparams.get_optimizer(model, training_hparams)
    lr_scheduler = Hparams.get_lr_scheduler(optimizer, training_hparams)
    loss_criterion = Hparams.get_loss_criterion(training_hparams)
    
    #implement early stopping instead
    print("Started training ...")
    iter = 0
    max_iter = dataloaderhelper.epochs_to_iter(training_hparams.num_epoch)
    running_loss = 0.0
    while iter < max_iter:
        for data in trainloader:
            #shuffle data for each epoch,
            #usually done by setting shuffle = True in the dataloader
            #but not in our case since we have a custom one
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = resnet20model(inputs)
            loss = loss_criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if iter % 100 == 0:
                print(f'[{iter}, {iter + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
                
                """#check early_stopping
                val_loss = get_val_loss(model, valloader, loss_criterion)
                if early_stopper.early_stop_val_loss(val_loss):
                    print("Stopped early")
                    break
                """
            lr_scheduler.step()
            iter += 1

def get_val_loss(model, valloader, loss_criterion):
    with torch.no_grad():
        cumulated_loss = 0
        for data in valloader:
            images, labels = data
            outputs = model(images)
            loss = loss_criterion(outputs, labels)
            cumulated_loss += loss.item()
            
    return cumulated_loss

def imp(model, training_hparams, pruning_hparams, saving_models_path,
        trainloader, valloader,
         max_pruning_level=12, warmup_iter=0):
    #TODO: replace pruning level by early stopping
    #TODO: implement warum_up before rewind point
    #TODO: Add calculation of statistics
    #create rewind point
    rewind_point = Resnet_N_W(model.plan, model.initializer, model.outputs)
    rewind_point.load_state_dict(model.state_dict())
    #save initial model
    torch.save(model.state_dict(), saving_models_path / "resnet-0.pth")
    for L in range(max_pruning_level):
        #do training
        train(
            model,
            training_hparams,
            trainloader,
            valloader
        )
        #pruning
        model.prune(
            prune_ratio = pruning_hparams.pruning_ratio,
            method = pruning_hparams.pruning_method
        )
        torch.save(model.state_dict(), saving_models_path / ("resnet-" + str(L + 1) + ".pth"))
        #rewind
        model.rewind(rewind_point)

def calculate_stats():
    #TODO: implement
    pass

models_path = workdir / "models"
if not models_path.exists():
    models_path.mkdir(parents=True)

saving_models_path = models_path / "experiment1"
if not saving_models_path.exists():
    saving_models_path.mkdir(parents=True)

#initialize hyperparemeters
training_hparams = Hparams.TrainingHparams(num_epoch=1, milestone_steps=[2])
pruning_hparams = Hparams.PruningHparams()

import time
start = time.time()
imp(
    resnet20model,
    training_hparams,
    pruning_hparams,
    saving_models_path,
    trainloader,
    valloader,
    max_pruning_level = 3
    )
end = time.time()
print("Time of IMP:", end - start)