import pandas as pd
import os
import yaml
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # configure the working directory to the project root path
    with open("config.yaml", "r", encoding="utf8") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    os.chdir(conf["project_path"])

    trajectories = pd.read_csv(conf['generation']['result'])
    print(trajectories['station'].value_counts() / trajectories['station'].value_counts().sum())
    print(
        trajectories.loc[trajectories['event'] == 'charging', 'queuing'].describe(
            percentiles=[0.5, 0.6, 0.7, 0.8, 0.9]))
