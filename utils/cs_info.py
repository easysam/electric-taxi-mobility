import pandas as pd
import numpy as np
import utils.data_loader as data_loader
import datetime


def nearest_station_by_dt(items, pivot):
    # returns the date that should be used for station data
    return min(items, key=lambda x: abs(x - pivot)), np.argmin(list(map(lambda x: abs(x - pivot), items)))


def get_cs_info_by_ce(df_cs, dates, ce):
    date_i = nearest_station_by_dt(dates, ce['source_t'])[0].strftime('%Y-%m')
    df_cs_i = df_cs[date_i].reset_index(drop=True)
    return df_cs_i


def get_cs_info_by_date(df_cs, dates, the_date):
    date_i = nearest_station_by_dt(dates, the_date)[0].strftime('%Y-%m')
    df_cs_i = df_cs[date_i].reset_index(drop=True)
    return df_cs_i


if __name__ == '__main__':
    df_cs, dates = data_loader.load_cs()

    df_cs_i = get_cs_info_by_ce(df_cs, dates, datetime.datetime(2014, 7, 5))
    print(df_cs_i)
