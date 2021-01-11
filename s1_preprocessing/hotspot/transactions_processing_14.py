import os
import time
import yaml
import pandas as pd
import logging
import pickle
from utils import display, od_utils
from s1_preprocessing.hotspot.hotpots_discovery_utils import generate_cube_index, hotspots_discovery_meanshift,\
    cube_to_coordinate

if __name__ == '__main__':
    # configure the working directory to the project root path
    with open("../config.yaml", "r", encoding="utf8") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    os.chdir(conf["project_path"])

    # initialization
    display.configure_pandas()
    display.configure_logging()

    # 读取transactions数据
    path = r'data/transaction_201407.csv'
    transactions = pd.read_csv(path, parse_dates=['begin_time', 'end_time'], infer_datetime_format=True,
                               low_memory=False)

    # filter od in bbox
    filter_od = od_utils.filter_in_bbox(transactions)
    print('Shape of transactions that out of bbox:', transactions.shape[0] - filter_od.shape[0])
    # 将points分到cubes里
    filtered_od = generate_cube_index(filter_od)

    # 聚类
    logging.info('Clustering load event')
    start_time = time.time()
    load_clusters = hotspots_discovery_meanshift(filtered_od, event='load')
    print("--- Clustering using %s seconds ---" % (time.time() - start_time))

    logging.info('Clustering drop event')
    start_time = time.time()
    drop_clusters = hotspots_discovery_meanshift(filtered_od, event='drop')
    print("--- Clustering using %s seconds ---" % (time.time() - start_time))

    # After merge, records out of bbox and not in common is null in 8 tail columns
    df_to_dump = pd.merge(transactions, filter_od[['Licence', 'begin_time', 'original_x', 'original_y',
                                                   'original_cube', 'destination_x', 'destination_y',
                                                   'destination_cube', 'load_label', 'drop_label']],
                          on=['Licence', 'begin_time'], how='left', indicator=True)
    print(transactions.shape, df_to_dump.shape)

    df_to_dump.to_csv('data/od/fod_w_14f_hs.csv', index=False)

    # Add geodetic coordinates to hot spots
    arranged_load = [{'x': item['x'], 'y': item['y'],
                      'longitude': cube_to_coordinate(item['cube'], to_geodetic=True)[0],
                      'latitude': cube_to_coordinate(item['cube'], to_geodetic=True)[1],
                      'cube': item['cube'], 'hotpots': list(set(item['hotpots'])),
                      'cube_geodetic': [cube_to_coordinate(cube, to_geodetic=True) for cube in
                                        list(set(item['hotpots']))]
                      } for item in load_clusters]

    arranged_drop = [{'x': item['x'], 'y': item['y'],
                      'longitude': cube_to_coordinate(item['cube'], to_geodetic=True)[0],
                      'latitude': cube_to_coordinate(item['cube'], to_geodetic=True)[1],
                      'cube': item['cube'], 'hotpots': list(set(item['hotpots'])),
                      'cube_geodetic': [cube_to_coordinate(cube, to_geodetic=True) for cube in
                                        list(set(item['hotpots']))]
                      } for item in drop_clusters]
    # Add hotspots id to detail info
    for i in range(len(arranged_load)):
        arranged_load[i]['id'] = i

    for i in range(len(arranged_drop)):
        arranged_drop[i]['id'] = i

    logging.info('Dump load and drop clusters')
    with open('data/hotspot/f14p.list_of_dict', 'wb') as f:
        pickle.dump(arranged_load, f)
    with open('data/hotspot/f14d.list_of_dict', 'wb') as f:
        pickle.dump(arranged_drop, f)
