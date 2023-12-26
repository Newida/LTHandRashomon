import torch
import torchvision
import numpy as np

class TorchRandomSeed(object):
    """
    Class to be used when opening a with clause. On enter sets the random seed for torch based sampling, restores previous state on exit
    """
    def __init__(self, seed):
        self.seed = seed
        self.prev_random_state = None

    def __enter__(self):
        self.prev_random_state = torch.get_rng_state()
        torch.set_rng_state(torch.manual_seed(self.seed).get_state())

    def __exit__(self, exc_type, exc_value, exc_traceback):
        torch.set_rng_state(self.prev_random_state)

class DataLoaderHelper():
    """Custom shuffeling and custom splitting into train and validation set"""

    def __init__(self, split_seed, data_order_seed, datasethparams):
        self.datasethparams = datasethparams
        self.trainloader = None
        self.testloader = None
        self.validationloader = None
        self.trainloader = None
        self.data_order_generator = None
        self.data_split_generator = None
        self.split_seed = split_seed
        self.data_order_seed = data_order_seed
        
    def split_train_val(self, trainset, val_set_size=5000):
        generator = torch.Generator()
        generator.manual_seed(self.split_seed)
        self.data_split_generator = generator
        trainset, valset = torch.utils.data.random_split(trainset,
                [len(trainset)-val_set_size, val_set_size])
        return trainset, valset
    
    def get_trainset(self, safe_trainset_path, transform):
        trainset = torchvision.datasets.CIFAR10(root=safe_trainset_path, train=True,
                                        download=True, transform=transform)
        return trainset

    def get_testset(self, safe_testset_path, transform):
        testset = torchvision.datasets.CIFAR10(root=safe_testset_path, train=False,
                                       download=True, transform=transform)
        return testset
    
    def get_test_loader(self, testset):
        testloader = torch.utils.data.DataLoader(testset,
                                                  batch_size=self.datasethparams.batch_size,
                                         shuffle=False, num_workers=10)
        self.testloader = testloader
        return testloader
    
    def get_validation_loader(self, valset):
        valloader = torch.utils.data.DataLoader(valset,
                                                  batch_size=self.datasethparams.batch_size,
                                         shuffle=False, num_workers=10)
        self.valloader = valloader
        return valloader
    
    def get_train_loader(self, trainset):
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=self.datasethparams.batch_size,
                                         shuffle=True, num_workers=1)
        #setting num_workers to 1 to avoid Randomness during mulit-process data loading
        self.trainloader = trainloader
        #make data order determinisitc
        #1. get random_sampler of train loader
        random_sampler = trainloader.sampler
        #2. initialize a generator
        generator = torch.Generator()
        generator.manual_seed(self.data_order_seed)
        self.data_order_generator = generator
        #3. pass generator to sampler
        random_sampler.generator = generator
        return trainloader
    
    def reset_trainloader_generator(self, trainloader):
        random_sampler = trainloader.sampler
        generator = trainloader.sampler.generator
        generator.manual_seed(self.data_order_seed)
    
    def iter_to_epochs(self, num_iters):
         return num_iters / len(self.trainloader)
    
    def epochs_to_iter(self, num_epochs):
         return num_epochs * len(self.trainloader)
    
class EarlyStopper:
    def __init__(self, patience = 1, min_delta = 0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_val_loss = float('inf')

    def early_stop_val_loss(self, val_loss):
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.counter = 0
        elif val_loss > (self.min_val_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False