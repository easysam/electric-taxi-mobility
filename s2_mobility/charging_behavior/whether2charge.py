import os
import yaml
import argparse
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import xgboost as xgb
from sklearn import preprocessing, metrics
from utils import display, data_loader


def process_grp(od):
    if od.at[od.index[-1], 'traveled_after_charged'] > 250:
        od['grp'] = 0
    if od.at[od.index[-1], 'traveled_after_charged'] < 50:
        od['grp'] = 0
    if od.at[od.index[-1], 'seeking_duration'] > np.timedelta64(60, 'm'):
        od['grp'] = 0
    return od


def train_evaluate_xgb(od, load_model=False, _model_path=None):
    msk = np.random.rand(len(od)) < 0.8
    train_set = od[msk]
    test_set = od[~msk]
    feature_columns = ['time_of_day', 'min_dis', 'max_dis', 'mean_dis', 'mid_dis', 'traveled_after_charged']
    train_X = train_set[feature_columns].to_numpy()
    train_y = train_set['to_charge']
    test_X = test_set[feature_columns].to_numpy()
    test_y = test_set['to_charge']

    gbm = xgb.XGBClassifier(verbosity=1, max_depth=10, learning_rate=0.01, n_estimators=500, scale_pos_weight=10)
    if load_model:
        gbm.load_model(_model_path)
    else:
        gbm.fit(train_X, train_y)
    predict_y = gbm.predict(test_X)

    recall = metrics.recall_score(test_y, predict_y)
    precision = metrics.precision_score(test_y, predict_y)
    accuracy = metrics.accuracy_score(test_y, predict_y)

    print("recall: %.2f, precision: %.2f, accuray: %.2f" % (recall, precision, accuracy))
    return test_y, predict_y, gbm


if __name__ == '__main__':
    display.configure_logging()
    display.configure_pandas()
    tqdm.pandas()
    np.random.seed(0)
    parser = argparse.ArgumentParser(description="Whether2Charge (XGBoost)")
    parser.add_argument('--load_model', action='store_true')
    args = parser.parse_args()
    # configure the working directory to the project root path
    with open("../../config.yaml", "r", encoding="utf8") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    os.chdir(conf["project_path"])

    cols = ['id', 'o_t', 'd_t', 'traveled_after_charged', 'to_charge', 'seeking_duration',
            'min_dis', 'mid_dis', 'mean_dis', 'max_dis']
    df_od = data_loader.load_od(scale='full', with_feature=True)[cols]
    df_od.dropna(axis=0, subset=['traveled_after_charged'], inplace=True)
    df_od.reset_index(drop=True, inplace=True)

    # make group for successive od that following a charging event and followed by a charging event. For example:
    # od_info, to_charge, grp
    #        .......
    # .......,      True, 1
    # .......,     False, 2
    # .......,     False, 2
    # .......,      True, 2
    # .......,     False, 3
    #        .......
    df_od['grp'] = (~df_od['to_charge'] & df_od['to_charge'].shift() & (df_od['id'] != df_od['id'].shift())).cumsum()

    df_od = df_od.groupby('grp').apply(process_grp)
    df_od = df_od.loc[0 != df_od['grp']]
    df_od.reset_index(drop=True, inplace=True)
    df_od['to_charge'] = df_od['to_charge'].astype(int)
    df_od['time_of_day'] = df_od['d_t'].dt.hour + df_od['d_t'].dt.minute / 60 + df_od['d_t'].dt.second / 3600

    # data standardization
    scaler = preprocessing.StandardScaler()
    transformed_data = scaler.fit_transform(df_od[['traveled_after_charged', 'min_dis', 'mid_dis', 'mean_dis',
                                                   'max_dis', 'time_of_day']])
    transformed_data = pd.DataFrame(transformed_data, columns=['traveled_after_charged', 'min_dis', 'mid_dis',
                                                               'mean_dis', 'max_dis', 'time_of_day'])
    transformed_data['o_t'] = df_od['o_t']
    transformed_data['id'] = df_od['id']
    transformed_data['to_charge'] = df_od['to_charge']

    model_path = conf['mobility']['charge']['whether_xgb']
    test_y, predict_y, gbm = train_evaluate_xgb(transformed_data, load_model=args.load_model, _model_path=model_path)
    gbm.save_model(model_path)
    with open(conf['mobility']['charge']['whether_xgb_scaler'], 'wb') as f:
        pickle.dump(scaler, f)
    # print(test_y)
    # print(predict_y)
