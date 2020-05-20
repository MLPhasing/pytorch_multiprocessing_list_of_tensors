import os
import argparse
import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.distributed as dist
import time

class FakeDataset(Dataset):
    def __init__(self):
        start_data = time.time()
        self.list = torch.rand((20480, 3, 224, 224))
        print("Data setup took: {}s".format(time.time() - start_data))
        self.length = 20480
        print("size in nbytes: {}".format(self.list.numel() * self.list.element_size()))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.list[idx], torch.tensor(0)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer_1 = torch.nn.Conv2d(3, 20, kernel_size=3)
        self.layer_2 = torch.nn.Conv2d(20, 50, kernel_size=3)
        self.layer_3 = torch.nn.Conv2d(50, 5, kernel_size=3)
        self.layer_4 = torch.nn.Linear(5 * 218 * 218, 10)

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = self.layer_3(self.layer_2(self.layer_1(x)))
        x = x.view(batch_size, -1)
        x = self.layer_4(x)
        return torch.log_softmax(x, dim=1)

def train(world_size, rank, num_epochs):
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    # In this simple case, the GPU index equals the rank
    gpu_index = rank

    torch.manual_seed(0)
    model = ConvNet()
    torch.cuda.set_device(gpu_index)
    model.cuda(gpu_index)
    batch_size = 128
    criterion = nn.functional.nll_loss
    lr = 1e-3 * world_size  # Larger world_size implies larger batches -> scale LR
    optimizer = torch.optim.Adam(model.parameters(), lr)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu_index])

    train_dataset = FakeDataset()
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=10,
        pin_memory=True,
        sampler=train_sampler
    )

    start = time.time()
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        print("Epoch: {}".format(epoch))
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print("Training completed in: {}s".format(time.time() - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--epochs', default=10, type=int)

    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8889'
    train(args.world_size, args.rank, args.epochs)
