import pandas as pd
import os
import yaml
import matplotlib.pyplot as plt
from utils import display

if __name__ == '__main__':
    display.configure_pandas()
    # configure the working directory to the project root path
    with open("config.yaml", "r", encoding="utf8") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    os.chdir(conf["project_path"])

    prediction = pd.read_parquet('result/generation/transition_gt')
    print((prediction['station'].value_counts() / prediction['station'].value_counts().sum()).head(30))
    print((prediction['station'].value_counts().iloc[:10].index < 10).sum())
    print((prediction['station'].value_counts().iloc[:20].index < 20).sum())
    print((prediction['station'].value_counts().iloc[:30].index < 30).sum())
    print((prediction['station'].value_counts().iloc[:30].index < 50).sum())
    print((prediction['station'].value_counts().iloc[:150].index < 150).sum())
    # print(
    #     prediction.loc[prediction['event'] == 'charging', 'queuing'].describe(
    #         percentiles=[0.5, 0.6, 0.7, 0.8, 0.9]))

    # trajectories = pd.read_csv(
    #     r'C:\Users\hkrep\Documents\research\城市感知大数据计算\电车充电预测\ET Charging Events\201706\part-r-00000', header=None)
    # print(trajectories[8].value_counts())
    # print('Total {} stations'.format(trajectories[8].nunique()))
    # print('Total {} vehicles'.format(trajectories[0].nunique()))
    # trajectories = []
    # for _ in range(1, 9):
    #     path = r'C:\Users\hkrep\Documents\research\城市感知大数据计算\电车充电预测\ET Charging Events\20170'+str(_)+'\part-r-00000'
    #     trajectories.append(pd.read_csv(path, header=None))
    # trajectories = pd.concat(trajectories)
    # print(trajectories[8].value_counts())
    # print('Total {} stations'.format(trajectories[8].nunique()))
    # print('Total {} vehicles'.format(trajectories[0].nunique()))

    # print(prediction.loc[prediction['event']=='charging'])
