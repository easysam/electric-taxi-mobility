from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np
import math
import time
import logging


# 输入：坐标点
# 输出：簇的中心的集、坐标点的label
def meanshift_cluster(X, plot=True):
    # X is points
    # 带宽，也就是以某个点为核心时的搜索半径
    bandwidth = estimate_bandwidth(X, quantile=0.003, n_samples=10000, n_jobs=-1)
    print('bandwidth is:', bandwidth)
    # 设置均值偏移函数
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1)
    # 训练数据
    ms.fit(X)
    # 每个点的标签
    labels = ms.labels_
    # print('labels of each points: ', labels)
    # 簇中心的点的集合
    cluster_centers = ms.cluster_centers_
    # 总共的标签分类
    labels_unique = np.unique(labels)
    # 聚簇的个数，即分类的个数
    n_clusters_ = len(labels_unique)

    if plot:
        plt.figure()
        plt.clf()
        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for k, col in zip(range(n_clusters_), colors):
            # 根据labels中的值是否等于k，重新组成一个True、False的数组
            my_members = labels == k
            cluster_center = cluster_centers[k]
            # X[my_members, 0] 取出my_members对应位置为True的值的横坐标
            plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
            plt.plot(cluster_center[0], cluster_center[1], '.', markerfacecolor=col,
                     markeredgecolor='k', markersize=4)
        # plt.axis([113.7704448, 114.3502372, 22.46477315, 22.79061185])
        plt.axis([0, 600, 0, 399])
        # 获取到当前坐标轴信息
        ax = plt.gca()
        # 将X坐标轴移到上面
        ax.xaxis.set_ticks_position('top')
        # 反转Y坐标轴
        ax.invert_yaxis()
        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.show()
    return cluster_centers, labels


def plot_points(transactions, coordinates='geodetic', hour=-1):
    plt.figure()
    plt.clf()
    if hour == -1:
        if coordinates == 'geodetic':
            plt.plot(transactions['original_log'], transactions['original_lat'], '.')
            plt.title('all load event')
        elif coordinates == 'cube':
            plt.plot(transactions['original_x'], transactions['original_y'], '.')
            plt.title('all load event')
            # 获取到当前坐标轴信息
            ax = plt.gca()
            # 将X坐标轴移到上面
            ax.xaxis.set_ticks_position('top')
            # 反转Y坐标轴
            ax.invert_yaxis()
        plt.show()
    else:
        if coordinates == 'geodetic':
            plt.plot(
                transactions.loc[(transactions.begin_time.dt.hour == hour) & (transactions.begin_time.dt.minute > 0)
                                 & (transactions.begin_time.dt.minute < 5), 'original_log'],
                transactions.loc[(transactions.begin_time.dt.hour == hour) & (transactions.begin_time.dt.minute > 0)
                                 & (transactions.begin_time.dt.minute < 5), 'original_lat'], '.')
            plt.title('Load event in %d clock(geodetic)' % hour)
        elif coordinates == 'cube':
            plt.plot(
                transactions.loc[(transactions.begin_time.dt.hour == hour) & (transactions.begin_time.dt.minute > 0)
                                 & (transactions.begin_time.dt.minute < 5), 'original_x'],
                transactions.loc[(transactions.begin_time.dt.hour == hour) & (transactions.begin_time.dt.minute > 0)
                                 & (transactions.begin_time.dt.minute < 5), 'original_y'], '.')
            plt.title('Load event in %d clock(cube)' % hour)
            # 获取到当前坐标轴信息
            ax = plt.gca()
            # 将X坐标轴移到上面
            ax.xaxis.set_ticks_position('top')
            # 反转Y坐标轴
            ax.invert_yaxis()
        plt.show()


def generate_cube_index(df_od, m=400, n=800):
    # geodetic coordinates to cube index
    bl_lng = 113.764635
    bl_lat = 22.454727
    tr_lng = 114.608972
    tr_lat = 22.842654
    X = (tr_lng - bl_lng) / n
    Y = (tr_lat - bl_lat) / m
    df_od['original_x'] = np.floor((df_od['original_log'] - bl_lng) / X).astype(int)
    df_od['original_y'] = np.floor((tr_lat - df_od['original_lat']) / Y).astype(int)
    df_od['original_cube'] = df_od['original_x'] + df_od['original_y'] * n
    df_od['destination_x'] = np.floor((df_od['destination_log'] - bl_lng) / X).astype(int)
    df_od['destination_y'] = np.floor((tr_lat - df_od['destination_lat']) / Y).astype(int)
    df_od['destination_cube'] = df_od['destination_x'] + df_od['destination_y'] * n
    return df_od


def to_integer_cube(original_x, original_y, original='geodetic'):
    # geodetic coordinates to cube index
    m = 400
    n = 800
    if original == 'geodetic':
        bl_lng = 113.764635
        bl_lat = 22.454727
        tr_lng = 114.608972
        tr_lat = 22.842654
        X = (tr_lng - bl_lng) / n
        Y = (tr_lat - bl_lat) / m
        x = math.floor((original_x - bl_lng) / X)
        y = math.floor((tr_lat - original_y) / Y)
    elif original == 'cube':
        x = math.floor(original_x)
        y = math.floor(original_y)
    else:
        raise NotImplementedError
    cube = x + y * n
    return x, y, cube


# If to_geodetic is True, then return center geodetic coordinates
def cube_to_coordinate(cube, to_geodetic=False):
    # geodetic coordinates to cube index
    m = 400
    n = 800
    y = cube // n
    x = cube % n
    if to_geodetic:
        bl_lng = 113.764635
        bl_lat = 22.454727
        tr_lng = 114.608972
        tr_lat = 22.842654
        X = (tr_lng - bl_lng) / n
        Y = (tr_lat - bl_lat) / m
        x = (x + 0.5) * X + bl_lng
        y = tr_lat - (y + 0.5) * Y
    return x, y


def hotspots_discovery_meanshift(transactions, event='load'):
    # Load Event Cluster!
    # 准备聚类的数据
    # X = transactions.loc[(transactions.begin_time.dt.hour == hour) & (transactions.begin_time.dt.minute > 0)
    #                      & (transactions.begin_time.dt.minute < 5), ['original_lat', 'original_log']].values
    if 'load' == event:
        X = transactions[['original_x', 'original_y']].values
        label_name, cube_name = 'load_label', 'original_cube'
    elif 'drop' == event:
        X = transactions[['destination_x', 'destination_y']].values
        label_name, cube_name = 'drop_label', 'destination_cube'
    else:
        raise NotImplementedError
    cluster_centers, labels = meanshift_cluster(X)
    transactions[label_name] = labels
    print('In %s clustering: %d cluster in centers data, %d in labels data.' % (event, len(cluster_centers),
                                                                                transactions[label_name].nunique()))
    transactions.reset_index(drop=True, inplace=True)

    # 合并center point落在同一cube的classes：得到的lad_label_cubes的key是cube，value是cube对应的class id，但事实上没有cube重合的cluster
    # cube_vs_label = dict()
    # for k, point in enumerate(cluster_centers):
    #     x, y, cube = to_integer_cube(point[0], point[1], original='cube')
    #     if cube in cube_vs_label:
    #         print(k, 'is merged')
    #         transactions.loc[transactions[label_name] == k, label_name] = cube_vs_label[cube]
    #     else:
    #         cube_vs_label[cube] = k

    # 统计每个cluster的信息，包括 1、中心点的x, y, cube，2、hotpots包含的cube
    clusters = []
    # for i, k in enumerate(list(cube_vs_label.values())):
    for k in range(len(cluster_centers)):
        x, y, cube = to_integer_cube(cluster_centers[k][0], cluster_centers[k][1], original='cube')
        cluster = dict({'x': x, 'y': y, 'cube': cube})
        cluster['hotpots'] = transactions.loc[transactions[label_name] == k, cube_name].tolist()
        cluster['distribution'] = len(transactions.loc[transactions[label_name] == k].index) / len(transactions.index)
        clusters.append(cluster)
        # transactions.loc[transactions[label_name] == k, 'new'+label_name] = i

    return clusters


'''
该函数已被弃用
由于scikit-learn中DBSCAN算法的实现需要算距离矩阵（不明），O（n2）复杂度
'''


def hotspots_by_dbscan(transactions, event='load'):
    if 'load' == event:
        X = transactions[['original_x', 'original_y']].values
        label_name = 'load_label'
        cube_name = 'original_cube'
    elif 'drop' == event:
        X = transactions[['destination_x', 'destination_y']].values
        label_name = 'drop_label'
        cube_name = 'destination_cube'
    else:
        raise NotImplementedError
    scaler = StandardScaler()
    x_standard = scaler.fit_transform(X)
    # 此处出现问题，当数据量大的时候，下面的dbscan的实现会占用所有内存，因此此函数弃用
    print('Start clustering.')
    start_time = time.time()
    db = DBSCAN(eps=0.3, min_samples=10, n_jobs=2).fit(x_standard)
    print("--- %s seconds ---" % (time.time() - start_time))
    print('Clustering completed.')
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels))


def generate_data_for_elki(transactions, event='load'):
    if 'load' == event:
        X = transactions[['original_log', 'original_lat']].values
        label_name = 'load_label'
        cube_name = 'original_cube'
    elif 'drop' == event:
        X = transactions[['original_log', 'original_lat']].values
        label_name = 'drop_label'
        cube_name = 'destination_cube'
    else:
        raise NotImplementedError
    scaler = StandardScaler()
    x_standard = scaler.fit_transform(X)
    x_standard = pd.DataFrame(x_standard)
    transactions_path = r'C:\Users\hkrep\PycharmProjects\ChargingEventsExtraction\data\od\elki\\' + 'od.csv'
    target_path = r'C:\Users\hkrep\PycharmProjects\ChargingEventsExtraction\data\od\elki\\' + event + '_original.csv'
    transactions.to_csv(transactions_path, index=False)
    x_standard.to_csv(target_path, index=False, header=False)


if __name__ == '__main__':
    print(cube_to_coordinate(0, to_geodetic=True))
