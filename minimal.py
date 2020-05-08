import pytorch_lightning as pl
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from torch.utils.data import DataLoader
import argparse

class FakeDataset(Dataset):
    def __init__(self, use_numpy):
        self.use_numpy = use_numpy
        print('Using numpy: {}'.format(use_numpy))
        if use_numpy:
            self.list = np.random.random((20000, 3, 100, 100)).astype(np.float32)
            self.length = 20000
            print("size in nbytes: {}".format(self.list.nbytes))
        else:
            self.list = torch.rand((20000, 3, 100, 100))
            self.length = 20000
            print("size in nbytes: {}".format(self.list.numel() * self.list.element_size()))
                
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.list[idx], torch.tensor(0)


class LightningWrapper(pl.LightningModule):
    def __init__(self, use_numpy, num_workers):
        super(LightningWrapper, self).__init__()
        self.use_numpy = use_numpy
        self.num_workers = num_workers
        self.layer_1 = torch.nn.Linear(3 * 100 * 100, 10)

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        return torch.log_softmax(x, dim=1)

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        return {'loss': loss}
    
    def prepare_data(self):
        self.training_set = FakeDataset(use_numpy=self.use_numpy)

    def train_dataloader(self):
        print('Number of workers: {}'.format(self.num_workers))
        return DataLoader(self.training_set, batch_size=32, num_workers=self.num_workers)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reconstruct memory overflow')
    parser.add_argument('--numpy', action='store_true')
    parser.add_argument('--num_workers', type=int, default='6')
    args = parser.parse_args()
    lightning_model = LightningWrapper(args.numpy, args.num_workers)
    trainer = pl.Trainer(gpus=1, distributed_backend='ddp', num_sanity_val_steps=0)
    trainer.fit(lightning_model)
