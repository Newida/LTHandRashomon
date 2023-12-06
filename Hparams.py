import torch

#hyper parameters taken from unmaksing the lth 
#training hyperparameters
#optimizer_name='sgd'
#momentum=0.9
#milestone_steps='80ep,120ep'
#lr=0.1
#gamma=0.1
#weight_decay=1e-4
#training_steps='160ep'
#data_order_seed = 0
#pruning hyperparemeters
#pruning_fraction = 0.2
#dataset hyperparameters
#batch_size = 128
#model hyperparameters
#model_initializer='kaiming_normal'
#batchnorm_init='uniform' TODO: What does that mean? How is this implemented by frankle? 
#rewind_points = [0, 250, 2000]
#loss = CrossEntropyLoss

class Hparams(object):
    """Collection of all hyperparameters used during training."""
    class DatasetHparams():
        def __init__(self, batch_size = 128) -> None:
            self.batch_size = batch_size
        #TODO: add missing:
        # transformation_seed
        # data_augmentation methods
        # otherstuff see online
    
    class TrainingHparams():
        def __init__(self, optimizer_name = "sgd", lr=0.1,
                      training_steps = "160ep", data_order_seed = None,
                       momentum = 0.9, gamma = 0.1, 
                       weight_decay=1e-4, loss_criterion = "crossentropy",
                         num_epoch = 160,
                         milestone_steps = [80, 120]) -> None:
            #TODO: find out what: nosterov is
            #TODO: replace num_epoch by early stopping
            #TODO: add missing hparams: 
            #training_steps, neserov_momentum, milsetone_steps, warmup_steps
            self.optimizer_name = optimizer_name
            self.lr = lr
            self.training_steps = training_steps
            self.data_order_seed = data_order_seed
            self.momentum = momentum
            self.weight_decay = weight_decay
            self.loss_criterion = loss_criterion
            self.num_epoch = num_epoch
            self.milestone_steps = milestone_steps #at which epoch to drop the learning rate
            #measured in epochs
            self.gamma = gamma #the amount the learning rate drops when reaching a milestone
    
    class PruningHparams():
        def __init__(self, pruning_ratio = 0.2, method = "l1") -> None:
            self.pruning_ratio = pruning_ratio
            self.pruning_method = method
             
    @staticmethod
    def get_optimizer(model, trainingHparams):
               if trainingHparams.optimizer_name == "sgd":
                    return torch.optim.SGD(
                         model.parameters(),
                         lr = trainingHparams.lr,
                         momentum = trainingHparams.momentum,
                         weight_decay = trainingHparams.weight_decay or 0
                    )
               if trainingHparams.optimizer_name == "adam":
                    return torch.optim.Adam(
                         model.parameters(),
                         lr = trainingHparams.lr,
                         weight_decay = trainingHparams.weight_decay or 0
                    )
               else:
                    raise ValueError("No such optimizer: choose either sgd or adam")
               
    @staticmethod
    def get_loss_criterion(trainingHparams):
          if trainingHparams.loss_criterion == "crossentropy":
               return torch.nn.CrossEntropyLoss()
          if trainingHparams.loss_criterion == "mse":
               return torch.nn.MSELoss()
          else:
               raise ValueError("No such loss: choose either crossentropy or mse")
          
    @staticmethod
    def get_lr_scheduler(optimizer, trainingHparams):
         return torch.optim.lr_scheduler.MultiStepLR(
               optimizer,
               milestones=trainingHparams.milestone_steps,
               gamma=trainingHparams.gamma
               )
