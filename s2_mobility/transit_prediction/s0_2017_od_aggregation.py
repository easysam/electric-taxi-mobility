import os
import yaml
import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from utils import display

if __name__ == '__main__':
    display.configure_logging()
    display.configure_pandas()
    # configure the working directory to the project root path
    with open("../../config.yaml", "r", encoding="utf8") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    os.chdir(conf["project_path"])
    # Hyper-parameter
    col_name = {"O_longitude": "original_log", "O_latitude": "original_lat", "D_longitude": "destination_log",
                "D_latitude": "destination_lat", "O_date_time": "begin_time", "D_date_time": "end_time"}

    # aggregate
    dd_od = dd.read_csv("data/od/201706/realiable_OD/*", include_path_column=True)
    dd_od = dd_od.rename(columns=col_name)
    dd_od = dd_od.drop(["O_difference", "D_difference"], axis=1)
    with ProgressBar():
        df_od = dd_od.compute()
    df_od["id"] = df_od["path"].str.rsplit('/', n=1, expand=True)[1]
    df_od.drop("path", axis=1, inplace=True)
    df_od.to_csv("data/od/201706_od.csv", index=False)

    _2017_et = pd.read_csv("data/201706_et_list_(license_is_Ecar)", )
    _2017_et = _2017_et.loc[_2017_et["is_Ecar"], "license"].to_list()
    od_files = ["data/od/201706/realiable_OD/" + vehicle for vehicle in _2017_et]
    dd_et_od = dd.read_csv(od_files, include_path_column=True)
    dd_et_od = dd_et_od.rename(columns=col_name)
    dd_et_od = dd_et_od.drop(["O_difference", "D_difference"], axis=1)
    with ProgressBar():
        df_et_od = dd_et_od.compute()
    df_et_od["id"] = df_et_od["path"].str.rsplit('/', n=1, expand=True)[1]
    df_et_od.drop("path", axis=1, inplace=True)
    df_et_od.to_csv("data/od/201706_et_od.csv", index=False)
