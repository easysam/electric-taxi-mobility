import os
import utils.data_loader as data_loader
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # show charging event distribution among time of day
    ce = data_loader.load_ce(version='v5_30min')
    common = data_loader.load_trajectory_od_intersection()
    ce = ce.loc[ce['licence'].isin(common)].reset_index(drop=True)
    ce['time_of_day'] = ce['start_charging'].dt.hour + ce['start_charging'].dt.minute / 60
    ce.plot(y='time_of_day', kind='hist', bins=24)
    plt.show()
