# location in the old project: preference_prediction_preprocessing/make_where_charge_dataset_v2.ipynb
import os
import yaml
import datetime
import pandas as pd
from tqdm import tqdm
from utils import data_loader, display


def fetch_departure_info(ce, df_od):
    """
    Find departure drop-off event of each charging event
    :param ce: single charging event, not a charging event data set
    :param df_od: transaction data set
    :return: departure for the charging event
    """
    begin = df_od['id'].searchsorted(ce['id'])
    end = df_od['id'].searchsorted(ce['id'], side='right')
    od_for_id_in_ce = df_od.iloc[begin:end]
    offset = od_for_id_in_ce['d_t'].searchsorted(ce['start_charging']) - 1
    # 如果该ce的licence不存在早于该ce的od，则返回空值。loc<0即为不存在。
    if offset < 0:
        return pd.Series(index=['d_t', 'd_l', 'traveled_after_charged', 'seeking_duration',
                                'mid_dis', 'min_dis', 'max_dis', 'mean_dis'])
    else:
        # traveled_after_charged可能是空的，其他的在这里都是非空的
        return od_for_id_in_ce.loc[od_for_id_in_ce.index[offset], ['d_t', 'd_l', 'traveled_after_charged',
                                                                   'seeking_duration', 'mid_dis', 'min_dis', 'max_dis',
                                                                   'mean_dis'] + [str(i) for i in range(23)]]


def account_charge_events(row, df_ce_feature):
    hs_begin = df_ce_feature['source_d_l'].searchsorted(row['id'])
    hs_end = df_ce_feature['source_d_l'].searchsorted(row['id'], side='right')
    hs_ce = df_ce_feature.iloc[hs_begin: hs_end]
    hs_distribution = hs_ce['cs_name'].value_counts()
    return hs_distribution


if __name__ == '__main__':
    tqdm.pandas()
    display.configure_pandas()
    display.configure_logging()

    # configure the working directory to the project root path
    with open("../../config.yaml", "r", encoding="utf8") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    os.chdir(conf["project_path"])

    # Load data set
    # od with feature (od_with_traveled_v?.csv) is created by whether_to_charger_data_preparation.py
    df_od = data_loader.load_od(scale='full', with_feature=True)
    df_ce = data_loader.load_ce(version='v5_30min')
    # If "id" columns is already exists, following code should be commented.
    df_ce.rename(columns={'licence': 'id'}, inplace=True)
    # drop_hss: drop-off hotspots
    drop_hss = data_loader.drop_clusters()
    drop_hss = pd.DataFrame.from_dict(drop_hss)
    # df_cs: charging station info, i.e., location, charger number ... for each charging station.
    df_cs, dates = data_loader.load_cs(date=datetime.datetime(2014, 7, 1))
    # Drop fake CS, these fake stations are clustered from ET trajectories by Wang Guang, which miss charger number info
    df_cs = df_cs.loc[~df_cs['cs_name'].isin(['LJDL', 'E04', 'BN0002', 'F11', 'S1', 'S2', 'F12', 'F13', 'F15'])]\
        .reset_index(drop=True)

    # Extract info for each charging event when the drivers make decision of charging station
    print("Extracting info for charging event...")
    departure_info = df_ce.progress_apply(fetch_departure_info, axis=1, args=(df_od,))
    departure_info.rename(columns={'d_t': 'source_d_t', 'd_l': 'source_d_l'}, inplace=True)
    df_ce = pd.concat([df_ce, departure_info], axis=1)

    # GROUND TRUTH
    # 去掉不能用来统计真值和作为训练数据的ce：没有出发信息，because multiple CEs have same departure info
    df_ce.drop_duplicates(subset=['id', 'source_d_t'], keep='first', inplace=True)
    # Constraint 1: seeking_duration < 1 hour, otherwise the station choice is irrelevant to the departure info
    df_ce = df_ce.loc[df_ce['seeking_duration'].dt.total_seconds() < 7200]
    # Construct ground truth, i.e., statistically count station distribution for each drop hotspot.
    df_ce.sort_values('source_d_l', inplace=True)
    d_hs_distribution = drop_hss.apply(account_charge_events, axis=1, args=(df_ce,))
    d_hs_distribution.set_index(drop_hss['id'], drop=True, inplace=True)
    # Constraint 2: records amount > 40
    d_hs_distribution = d_hs_distribution.loc[d_hs_distribution.sum(axis=1) > 40]
    # 去掉不能用来统计和训练的ce：没有出发热区的对应真值分布、没有traveled feature
    df_ce = df_ce.loc[df_ce['source_d_l'].isin(d_hs_distribution.index)]
    df_ce.dropna(subset=['traveled_after_charged'], inplace=True)
    df_ce.sort_values(['source_d_l', 'source_d_t'], inplace=True)
    # Attach ground truth the charging event data set, to construct the labels
    gt = pd.merge(df_ce['source_d_l'], d_hs_distribution, left_on='source_d_l', right_index=True)
    gt = gt[df_cs['cs_name']]
    # Transform number to probability distribution
    gt = gt.fillna(0).divide(gt.sum(axis=1), axis=0)
    # save GROUND TRUTH to local
    path = 'result/mobility/charge/whether/train_set'
    os.makedirs(path, exist_ok=True)
    gt.to_csv(os.path.join(path, "all_formatted_annotation_distribution_v3.csv"), index=False)

    # FEATURE
    # 删掉多余feature, 为了按站给record，feature要作为 identifiers set.
    df_ce.drop(
        ['id', 'arrival_time', 'begin_time', 'start_charging', 'end_time', 'waiting_duration', 'charging_duration',
         'cs_name', 'valid', 'seeking_duration'], axis=1, inplace=True)
    # 将cs distance改为stack形式，ce_feature从逐ce变为逐ce逐cs
    ce_feature = df_ce.melt(
        id_vars=['source_d_l', 'max_dis', 'mean_dis', 'mid_dis', 'min_dis', 'traveled_after_charged', 'source_d_t'],
        var_name='cs_index', value_name='distance')
    # 按照 cs_index 顺序排序，与gt ground truth 的cs charging station顺序对应
    ce_feature['cs_index'] = ce_feature['cs_index'].astype(int)
    ce_feature = ce_feature.sort_values(['source_d_l', 'source_d_t', 'cs_index'])
    ce_feature['weekday'] = ce_feature['source_d_t'].dt.weekday < 5
    ce_feature['time_of_day'] = ce_feature['source_d_t'].dt.hour + ce_feature['source_d_t'].dt.minute / 60
    ce_feature.drop('source_d_t', axis=1, inplace=True)
    ce_feature = pd.merge(ce_feature, df_cs['chg_points'], left_on='cs_index', right_index=True, how='left')
    # save feature to local
    ce_feature.to_csv("result/mobility/charge/where/where_charge_feature.csv", index=False)
