import os
import yaml
import pandas as pd
from utils import display


if __name__ == '__main__':
    # configure the working directory to the project root path
    with open("config.yaml", "r", encoding="utf8") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    os.chdir(conf["project_path"])

    display.configure_pandas()
    ce17 = pd.read_parquet('data/charging_event/charging_event_30min_200m_531station.parquet')
    cs_info = pd.read_csv(conf['cs']['val_cdzs'])

    real_cs = ce17['s_idx'].value_counts().loc[ce17['s_idx'].value_counts() > 30]
    new_cs = pd.concat([cs_info.loc[real_cs.index], real_cs], axis=1)
    new_cs.reset_index(drop=True, inplace=True)
    new_cs['chg_points'] = new_cs['dcNum'] + new_cs['acNum']
    print(new_cs.head(30))
    # new_cs = new_cs[['lng', 'lat', 'chg_points']]
    new_cs.to_csv('s3_generation/cs_program1/cs_info.csv', index=False)

