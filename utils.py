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

    def __init__(self, split_seed, data_order_seed, val_seed, test_seed, datasethparams):
        self.datasethparams = datasethparams
        self.trainloader = None
        self.testloader = None
        self.validationloader = None
        self.trainloader = None
        self.data_order_generator = None
        self.data_split_generator = None
        self.test_seed = test_seed
        self.val_seed = val_seed
        self.split_seed = split_seed
        self.data_order_seed = data_order_seed
        
    def split_train_val(self, trainset, val_set_size=5000):
        generator = torch.Generator()
        generator.manual_seed(self.split_seed)
        self.data_split_generator = generator
        trainset, valset = torch.utils.data.random_split(trainset,
                [len(trainset)-val_set_size, val_set_size], generator=generator)
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
        generator = torch.Generator()
        generator.manual_seed(self.test_seed)
        testloader = torch.utils.data.DataLoader(testset,
                                                  batch_size=self.datasethparams.batch_size,
                                         shuffle=False, num_workers=1, generator = generator)
        self.testloader = testloader
        return testloader
    
    def get_validation_loader(self, valset):
        generator = torch.Generator()
        generator.manual_seed(self.val_seed)
        valloader = torch.utils.data.DataLoader(valset,
                                                  batch_size=self.datasethparams.batch_size,
                                         shuffle=False, num_workers=1, generator=generator)
        self.validationloader = valloader
        return valloader
    
    def get_train_loader(self, trainset):
        generator = torch.Generator()
        generator.manual_seed(self.data_order_seed)
        self.data_order_generator = generator
        random_sampler = torch.utils.data.RandomSampler(data_source=trainset, generator=generator)
        trainloader = torch.utils.data.DataLoader(trainset,sampler=random_sampler,
                                                  batch_size=self.datasethparams.batch_size,
                                                  num_workers=1, generator=generator)
        self.trainloader = trainloader
        return trainloader
    
    def reset_valloader_generator(self):
        self.validationloader.generator.manual_seed(self.val_seed)

    def reset_trainloader_generator(self):
        random_sampler = self.trainloader.sampler
        generator = self.trainloader.sampler.generator
        generator.manual_seed(self.data_order_seed)
        #same as self.trainloader.generator.manual_seed(self.val_seed)
        #since we gave the same generator to
        #the dataloader and the sampler
    
    def iter_to_epochs(self, num_iters):
         return num_iters / len(self.trainloader)
    
    def epochs_to_iter(self, num_epochs):
         return num_epochs * len(self.trainloader)