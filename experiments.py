import pickle
import torch
from pathlib import Path
import torchvision
import torchvision.transforms as transforms
from resnet20 import Resnet_N_W
from Hparams import Hparams
from utils import EarlyStopper
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
    dataloaderhelper = utils.DataLoaderHelper(split_seed=0, data_order_seed=0, datasethparams=dataset_hparams)
    trainset = dataloaderhelper.get_trainset(data_path, transform)
    testset = dataloaderhelper.get_testset(data_path, transform)
    trainset, valset = dataloaderhelper.split_train_val(trainset)
    trainloader = dataloaderhelper.get_train_loader(trainset)
    testloader = dataloaderhelper.get_test_loader(testset)
    valloader = dataloaderhelper.get_validation_loader(valset)

early_stopper = EarlyStopper(patience=1, min_delta=0)

def e1_train_val_loss():
    #initialize network
    #1. Setup hyperparameters
    training_hparams = Hparams.TrainingHparams(num_epoch=162)
    pruning_hparams = Hparams.PruningHparams()
    model_structure, initializer, outputs = Resnet_N_W.get_model_from_name("resnet-20")
    model_hparams = Hparams.ModelHparams(
        model_structure, initializer, outputs, initialization_seed=0)
    #2. Setup model
    model = Resnet_N_W(model_hparams)
    #3. Train model
    _, all_stats = routines.train(device,
        model,
          0,
          dataloaderhelper,
          training_hparams,
          False
          )
    #4. Save model and statistics
    routines.save_experiment("e1",
                             dataset_hparams,
                             training_hparams,
                             pruning_hparams,
                             model_hparams,
                             [model],
                             [all_stats],
                             False)
    #5. Plot some results
    #TODO: next
    return all_stats

import time
start = time.time()
stats = e1_train_val_loss()
print(stats)
end = time.time()
print("Time of e1 with 1 workers:", end - start)

with open('outfile_withoutstats.pickle', 'wb') as fp:
    pickle.dump(stats, fp)

with open ('outfile_withoutstats.pickle', 'rb') as fp:
    itemlist = pickle.load(fp)