# Credits: This script is taken from https://github.com/kundajelab/deeplift/blob/master/deeplift/dinuc_shuffle.py
import os, pathlib, h5py
import numpy as np
from scipy import stats
import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader


def load_model_from_checkpoint(model, checkpoint_path):
    """Load PyTorch lightning model from checkpoint."""
    return model.load_from_checkpoint(checkpoint_path,
        model=model.model,
        criterion=model.criterion,
        optimizer=model.optimizer,
    )


def configure_optimizer(model, lr=0.001, weight_decay=1e-6, decay_factor=0.1, patience=5, monitor='val_loss'):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=decay_factor, patience=patience),
            "monitor": monitor,
        },
    }
    

def get_predictions(model, x, batch_size=100, accelerator='gpu', devices=1):
    trainer = pl.Trainer(accelerator=accelerator, devices=devices, logger=None)
    dataloader = torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=False) 
    pred = trainer.predict(model, dataloaders=dataloader)
    return np.concatenate(pred)


def evaluate_model(y_test, pred):
    pearsonr = calculate_pearsonr(y_test, pred)
    spearmanr = calculate_spearmanr(y_test, pred)
    #print("Test Pearson r : %.4f +/- %.4f"%(np.nanmean(pearsonr), np.nanstd(pearsonr)))
    #print("Test Spearman r: %.4f +/- %.4f"%(np.nanmean(spearmanr), np.nanstd(spearmanr)))
    print("  Pearson r: %.4f \t %.4f"%(pearsonr[0], pearsonr[1]))
    print("  Spearman : %.4f \t %.4f"%(spearmanr[0], spearmanr[1]))
    return pearsonr, spearmanr

def calculate_pearsonr(y_true, y_score):
    vals = []
    for class_index in range(y_true.shape[-1]):
        vals.append( stats.pearsonr(y_true[:,class_index], y_score[:,class_index])[0] )    
    return np.array(vals)
    
def calculate_spearmanr(y_true, y_score):
    vals = []
    for class_index in range(y_true.shape[-1]):
        vals.append( stats.spearmanr(y_true[:,class_index], y_score[:,class_index])[0] )    
    return np.array(vals)




class H5DataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size=128, stage=None, lower_case=False, transpose=False, downsample=None):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.x = 'X'
        self.y = 'Y'
        if lower_case:
            self.x = 'x'
            self.y = 'y'
        self.transpose = transpose
        self.downsample = downsample
        self.setup(stage)

    def setup(self, stage=None):
        # Assign train and val split(s) for use in DataLoaders
        if stage == "fit" or stage is None:
            with h5py.File(self.data_path, 'r') as dataset:
                x_train = np.array(dataset[self.x+"_train"]).astype(np.float32)
                y_train = np.array(dataset[self.y+"_train"]).astype(np.float32)
                x_valid = np.array(dataset[self.x+"_valid"]).astype(np.float32)
                if self.transpose:
                    x_train = np.transpose(x_train, (0,2,1))
                    x_valid = np.transpose(x_valid, (0,2,1))
                if self.downsample:
                    x_train = x_train[:self.downsample]
                    y_train = y_train[:self.downsample]
                self.x_train = torch.from_numpy(x_train)
                self.y_train = torch.from_numpy(y_train)
                self.x_valid = torch.from_numpy(x_valid)
                self.y_valid = torch.from_numpy(np.array(dataset[self.y+"_valid"]).astype(np.float32))
            _, self.A, self.L = self.x_train.shape # N = number of seqs, A = alphabet size (number of nucl.), L = length of seqs
            self.num_classes = self.y_train.shape[1]
            
        # Assign test split(s) for use in DataLoaders
        if stage == "test" or stage is None:
            with h5py.File(self.data_path, "r") as dataset:
                x_test = np.array(dataset[self.x+"_test"]).astype(np.float32)
                if self.transpose:
                    x_test = np.transpose(x_test, (0,2,1))
                self.x_test = torch.from_numpy(x_test)
                self.y_test = torch.from_numpy(np.array(dataset[self.y+"_test"]).astype(np.float32))
            _, self.A, self.L = self.x_train.shape
            self.num_classes = self.y_train.shape[1]
            
    def train_dataloader(self):
        train_dataset = TensorDataset(self.x_train, self.y_train) # tensors are index-matched
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True) # sets of (x, x', y) will be shuffled
    
    def val_dataloader(self):
        valid_dataset = TensorDataset(self.x_valid, self.y_valid) 
        return DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        test_dataset = TensorDataset(self.x_test, self.y_test)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False) 