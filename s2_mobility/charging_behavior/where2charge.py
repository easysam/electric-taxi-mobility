import os
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import statistics
import time
import yaml
from utils import display


class ChargingEventDataset(Dataset):
    def __init__(self, data, annotation, transform=None):
        self.data = data
        self.annotation = annotation
        self.transform = transform

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        data = torch.from_numpy(self.data.iloc[idx * 23: idx * 23 + 23, [1, 2, 3, 4, 5, 7, 8, 9, 10]].values)  # change
        label = torch.tensor(self.annotation[idx])
        sample = {'ce': data, 'annotation': label}

        if self.transform:
            sample = self.transform(sample)

        return data.float(), label.float()


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(9, 16)   # change
        self.l2 = torch.nn.Linear(16, 32)
        self.l3 = torch.nn.Linear(32, 16)
        self.l4 = torch.nn.Linear(16, 1)
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.relu3 = torch.nn.ReLU()
        self.relu4 = torch.nn.ReLU()

    # @torchsnooper.snoop()
    def forward(self, x):
        y = []
        for i in range(x.shape[1]):
            tmp_x = x[:, i].float()
            tmp_x = self.relu1(self.l1(tmp_x))
            tmp_x = self.relu2(self.l2(tmp_x))
            tmp_x = self.relu3(self.l3(tmp_x))
            tmp_x = self.relu4(self.l4(tmp_x))
            y.append(tmp_x)
        return torch.cat(y, -1)


def train(epoch, train_loader):
    model.train()
    loss_value = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data).to(device)
        softmax = torch.nn.Softmax(dim=1)
        output = softmax(output)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            # print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #            100. * batch_idx / len(train_loader), loss.item()))
            loss_value.append(loss.item())
    return statistics.mean(loss_value)


if __name__ == '__main__':
    # load data
    tqdm.pandas()
    display.configure_pandas()
    display.configure_logging()

    # configure the working directory to the project root path
    with open("../../config.yaml", "r", encoding="utf8") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    os.chdir(conf["project_path"])

    data_type = {"source_d_l": np.int16, "max_dis": np.float32, "mean_dis": np.float32, "mid_dis": np.float32,
                 "min_dis": np.float32, "traveled_after_charged": np.float32, "cs_index": np.int8,
                 "distance": np.float32, "weekday": np.int8, "time_of_day": np.float32, "chg_points": np.int8}
    df_all_data = pd.read_csv(conf["mobility"]["charge"]["where_feature"], dtype=data_type)
    print(df_all_data.dtypes)
    annotation = pd.read_csv(conf["mobility"]["charge"]["where_label"]).values

    ce_data_set = ChargingEventDataset(df_all_data, annotation)
    data_loader = DataLoader(ce_data_set, batch_size=32, shuffle=True, num_workers=2)

    # for i_batch, (data, label) in enumerate(data_loader):
    #     # observe 4th batch and stop.
    #     if i_batch == 3:
    #         break

    model = Net()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.5)

    # Save a time stamp
    print("Start training...")
    since = time.time()
    loss_value = []
    total_epoch = 100
    min_running_loss = 100
    for epoch in range(0, total_epoch):
        running_loss = 0.0
        epoch_start = time.time()
        running_loss += train(epoch, data_loader)
        loss_value.append(running_loss)
        m, s = divmod(time.time() - epoch_start, 60)
        print('epoch:', epoch, f'Training time: {m:.0f}m {s:.0f}s', 'avg_loss: ', running_loss)
        if running_loss < min_running_loss:
            torch.save(model.state_dict(), conf["mobility"]["charge"]["where_model"] + '.best')
            min_running_loss = running_loss
    plt.plot(loss_value)
    plt.show()
    # Show time consumption
    m, s = divmod(time.time() - since, 60)
    print(f'Total Time: {m:.0f}m {s:.0f}s\nModel was trained on {device}!')

    torch.save(model.state_dict(), conf["mobility"]["charge"]["where_model"] + str(total_epoch))
