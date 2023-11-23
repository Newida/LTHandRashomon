import torch
import torchvision

class DataLoaderHelper():
    """Custom shuffeling and custom splitting into train and validation set"""

    def __init__(self, split_seed, dataorder_seed, datasethparams):
        self.datasethparams = datasethparams
        self.split_seed = split_seed
        self.dataorder_seed = dataorder_seed
        self.split_generator = torch.Generator()
        self.split_generator.manual_seed(self.split_seed)
        self.train_generator = torch.Generator()
        self.train_generator.manual_seed(self.dataorder_seed)

    def split_train_val(self, trainset, val_set_size=5000):
        trainset, valset = torch.utils.data.random_split(trainset,
                [len(trainset)-val_set_size, val_set_size], generator=self.split_generator)
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
                                         shuffle=False, num_workers=2)
        return testloader
    
    def get_validation_loader(self, valset):
        return self.get_test_loader(valset)
    
    def get_train_loader(self, trainset):
        trainloader = TrainLoader(trainset, self.datasethparams.batch_size,
                                  self.train_generator)
        return trainloader


class TrainLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, train_generator, num_workers=2):
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_generator = train_generator
        self.sampler = ShuffleSampler(len(dataset), self.train_generator)
        super(TrainLoader, self).__init__(dataset, batch_size,sampler=self.sampler,
                                           num_workers=num_workers)

    def get_seed(self):
        return self.train_generator.initial_seed()
    
    def shuffle(self, seed=None):
        if seed is None:
            self.sampler.shuffle_dataorder(self.train_generator.initial_seed())
        else:
            self.sampler.shuffle_dataorder(seed)
        

class ShuffleSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, num_samples, generator):
        self.num_samples = num_samples
        self.generator = generator

    def __iter__(self):
        indices = torch.randperm(self.num_samples, generator=self.generator).tolist()
        return iter(indices)

    def __len__(self):
        return self.num_samples
    
    def shuffle_dataorder(self, seed):
        self.generator.manual_seed(seed)
        