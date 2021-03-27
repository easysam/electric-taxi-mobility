import os
import yaml
import pandas as pd
from utils import display


def cs_filter_by_ce(_cs, _ce, threshold=30, dc_name='dcNum', ac_name='acNum'):
    _cs_counts = _ce['s_idx'].value_counts().rename('freq')
    _filtered_cs = _cs_counts.loc[_cs_counts > threshold]
    _filtered_cs = pd.concat([_cs.loc[_filtered_cs.index], _filtered_cs], axis=1)
    _filtered_cs.reset_index(drop=True, inplace=True)
    _filtered_cs['chg_points'] = _filtered_cs[dc_name] + _filtered_cs[ac_name]
    return _filtered_cs


if __name__ == '__main__':
    # configure the working directory to the project root path
    with open("config.yaml", "r", encoding="utf8") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    os.chdir(conf["project_path"])
    display.configure_pandas()
    source_list = ['TC', 'cluster', 'pengchengchong', 'chongdianba']
    source = source_list[3]
    # Read files
    ce17 = pd.read_parquet(conf['ce'][source])
    cs_info = pd.read_csv(conf['cs']['raw'][source])
    # Process
    filtered_sorted_cs = cs_filter_by_ce(cs_info, ce17, threshold=30,
                                         dc_name='fastChargingPileCount',
                                         ac_name='slowChargingPileCount')
    filtered_sorted_cs.to_csv(conf['cs']['filtered_sorted'][source], index=False)
