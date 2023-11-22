Reproducibility:
Goal:
Given a seed for the splits and a seed for training: validation set and shuffling during training should stay the same given the seed
Strategy:
- Loading dataset:
-- define seed
-- split training data into training and validation
-- use training data as before
-- add early stopping using the validation dataset
- dont shuffle validation set and test set
- shuffle training data for each epoch by hand
Remarks:
- done by classes TrainDataloader and DataLoaderHelper

EarlyStopping:
Goal: Stop when minimum validation loss
Strategy:
- evaluate validation every 100 iterations as done in the paper
- safe network and its stats: 
Remarks:
- done by class EarlyStopping
- uses val_loss calculated in during training
- val_loss gets checked every 100 iterations