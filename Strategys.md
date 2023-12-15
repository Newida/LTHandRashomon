EarlyStopping:
Goal: Stop when minimum validation loss
Strategy:
- evaluate validation every 100 iterations as done in the paper
- safe network and its stats: 
Remarks:
- done by class EarlyStopping
- uses val_loss calculated in during training
- val_loss gets checked every 100 iterations