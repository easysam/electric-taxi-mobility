import torch
import os
import yaml
import time
import statistics
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils import display
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class TransitionDataset(Dataset):
    def __init__(self, data, annotation):
        self.data = data
        self.annotation = annotation

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data_item = torch.from_numpy(self.data[idx])
        annotation_item = torch.tensor(self.annotation[idx])
        return data_item.float(), annotation_item.float()


class Where2Charge(torch.nn.Module):
    def __init__(self):
        super(Where2Charge, self).__init__()
        self.l1 = torch.nn.Linear(16, 32)
        self.l2 = torch.nn.Linear(32, 1)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.relu1 = torch.nn.ReLU()

    # @torchsnooper.snoop()
    def forward(self, x):
        x = self.relu1(self.bn1(self.l1(x)))
        x = self.l2(x)
        return x


def train(epoch, train_loader):
    model.train()
    loss_value = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data).to(device)
        softmax = torch.nn.Softmax(dim=1)
        output = softmax(output)
        temp = torch.log(output)
        loss = criterion(temp, target)

        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            # print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #            100. * batch_idx / len(train_loader), loss.item()))
            loss_value.append(loss.item())
    return statistics.mean(loss_value)


if __name__ == '__main__':
    display.configure_pandas()
    display.configure_logging()

    parser = argparse.ArgumentParser(description='Transition Prediction Utility (XGBoost) Train')
    parser.add_argument('--task', type=str, default='p2d', choices=['p2d', 'd2p'])
    args = parser.parse_args()
    # configure the working directory to the project root path
    with open("../../config.yaml", "r", encoding="utf8") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    os.chdir(conf["project_path"])

    p2d_x = np.load(conf["mobility"]["transition"]["utility_xgboost"][args.task]["train_feature"])
    p2d_gt = pd.read_csv(conf["mobility"]["transition"]["utility_xgboost"][args.task]["train_gt"])

    p2d_train_x, p2d_val_x, _p2d_train_y, _p2d_val_y = train_test_split(p2d_x, p2d_gt, test_size=0.2)
    p2d_train_y, p2d_val_y = _p2d_train_y["rate"].to_numpy().reshape(-1, 1), _p2d_val_y["rate"].to_numpy().reshape(-1, 1)
    print(p2d_train_x.shape, p2d_train_y.shape)
    transition_dataset = TransitionDataset(p2d_train_x, p2d_train_y)
    data_loader = DataLoader(transition_dataset, batch_size=32, shuffle=True, num_workers=2)
    model = Where2Charge()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
    # Save a time stamp
    print("Start training...")
    since = time.time()
    loss_value = []
    total_epoch = 30
    min_running_loss = 100
    for epoch in range(0, total_epoch):
        running_loss = 0.0
        epoch_start = time.time()
        running_loss += train(epoch, data_loader)
        loss_value.append(running_loss)
        m, s = divmod(time.time() - epoch_start, 60)
        print('epoch:', epoch, f'Training time: {m:.0f}m {s:.0f}s', 'avg_loss: ', running_loss)
        if running_loss < min_running_loss:
            torch.save(model.state_dict(), 'test.best')
            min_running_loss = running_loss
    plt.plot(loss_value)
    plt.show()
    # Show time consumption
    m, s = divmod(time.time() - since, 60)
    print(f'Total Time: {m:.0f}m {s:.0f}s\nModel was trained on {device}!')

    torch.save(model.state_dict(), 'test.test')
