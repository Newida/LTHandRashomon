from typing import Any
import torch
import torchvision
import numpy as np
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

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

    def __init__(self, datasethparams):
        self.datasethparams = datasethparams
        self.testloader = None
        self.validationloader = None
        self.trainloader = None
        self.data_order_generator = None
        self.data_split_generator = None
        self.rngCrop = None
        self.rngRandomHflip = None
        
    def split_train_val(self, trainset, val_set_size=5000):
        generator = torch.Generator()
        generator.manual_seed(self.datasethparams.split_seed)
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
        generator.manual_seed(self.datasethparams.test_seed)
        testloader = torch.utils.data.DataLoader(testset,
                                                  batch_size=self.datasethparams.batch_size,
                                         shuffle=False, num_workers=1, generator = generator)
        self.testloader = testloader
        return testloader
    
    def get_validation_loader(self, valset):
        generator = torch.Generator()
        generator.manual_seed(self.datasethparams.val_seed)
        valloader = torch.utils.data.DataLoader(valset,
                                                  batch_size=self.datasethparams.batch_size,
                                         shuffle=False, num_workers=1, generator=generator)
        self.validationloader = valloader
        return valloader
    
    def get_train_loader(self, trainset):
        generator = torch.Generator()
        generator.manual_seed(self.datasethparams.train_seed)
        self.data_order_generator = generator
        random_sampler = torch.utils.data.RandomSampler(data_source = trainset, generator = generator)
        trainloader = torch.utils.data.DataLoader(trainset, sampler=random_sampler,
                                                  batch_size = self.datasethparams.batch_size,
                                                  num_workers = 1, generator = generator)
        self.trainloader = trainloader
        return trainloader
    
    def reset_valloader_generator(self):
        self.validationloader.generator.manual_seed(self.datasethparams.val_seed)

    def reset_testoader_generator(self):
        self.testloader.generator.manual_seed(self.datasethparams.test_seed)

    def reset_trainloader_generator(self):
        random_sampler = self.trainloader.sampler
        generator = self.trainloader.sampler.generator
        generator.manual_seed(self.datasethparams.train_seed)
        #same as self.trainloader.generator.manual_seed(self.train_seed)
        #since we gave the same generator to
        #the dataloader and the sampler
    
    def iter_to_epochs(self, num_iters):
         return num_iters / len(self.trainloader)
    
    def epochs_to_iter(self, num_epochs):
         return num_epochs * len(self.trainloader)
    
    def get_transform(self, augmented):
        if augmented:
            rngCrop = RandomCropTransform(self.datasethparams.rngCrop_seed)
            rngRandomHflip = RandomHflipTransform(self.datasethparams.rngRandomHflip_seed)
            self.rngCrop = rngCrop
            self.rngRandomHflip = rngRandomHflip
            return transforms.Compose([
                        self.rngCrop,
                        self.rngRandomHflip,
                        transforms.ToTensor(),
                        transforms.Normalize(self.datasethparams.mean, self.datasethparams.std)
                        ])
        else:
            return transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(self.datasethparams.mean, self.datasethparams.std)
                        ])

    
class RandomHflipTransform(object):

    def __init__(self, data_augmentation_seed) -> None:
        self.data_augmentation_seed = data_augmentation_seed
        self.generator = torch.Generator()
        self.generator.manual_seed(self.data_augmentation_seed)

    def __call__(self, image) -> Any:
        if torch.rand(1, generator=self.generator).item() < 0.5:
            image = TF.hflip(image)
        return image
        
    def reset_generator(self):
        self.generator.manual_seed(self.data_augmentation_seed)

class RandomCropTransform(object):

    def __init__(self, data_augmentation_seed) -> None:
        self.data_augmentation_seed = data_augmentation_seed
        self.generator = torch.Generator()
        self.generator.manual_seed(self.data_augmentation_seed)
        self.size = 32
        self.padding = 4

    def __call__(self, image):
        image = TF.pad(image, 4, fill=0)
        _, h, w = TF.get_dimensions(image)
        i = torch.randint(0, h - self.size + 1, generator=self.generator, size=(1,)).item()
        j = torch.randint(0, w - self.size + 1, generator=self.generator, size=(1,)).item()
        image = TF.crop(image, i, j, 32, 32)

        return image

    def reset_generator(self):
        self.generator.manual_seed(self.data_augmentation_seed)
