# coding=utf-8
import logging
import os
import time
import random
import re
import pandas as pd
import math
import sys
import csv
from multiprocessing import Pool
import multiprocessing
from road_match.config import cfg
from tqdm import tqdm
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
# 使用方法一得到两个经纬度点坐标之间的距离： LonA和LatA分别是A点的经度和纬度坐标
def getdistance1(LonA, LatA, LonB, LatB):
    radLng1 = LatA * math.pi / 180.0
    radLng2 = LatB * math.pi / 180.0
    a = radLng1 - radLng2
    b = (LonA - LonB) * math.pi / 180.0
    s = 2 * math.asin(math.sqrt(
    math.pow(math.sin(a / 2), 2) + math.cos(radLng1) * math.cos(radLng2) * math.pow(math.sin(b / 2), 2))) * 6378137  # 返回单位为米
    return s


# 使用方法二得到两个经纬度点坐标之间的距离： LonA和LatA分别是A点的经度和纬度坐标
def getdistance2(LonA, LatA, LonB, LatB):
    a = math.sin(LatA) * math.sin(LatB)  # 两点纬度乘积
    b = math.cos(LatA) * math.cos(LatB) * math.cos(LonA - LonB)
    # dis = math.acos(a+b)*6371004
    dis = math.acos(a + b) * 6378137
    return dis


# 事实证明使用第一种计算方法的距离值更加真实，因此我们将采用第一种方法来计算一个轨迹点到一个路段的距离。
# PCx,PCy是孤立查询点的经纬度坐标，A和B是路段的两个端点
def getPointWayDis(PAx, PAy, PBx, PBy, PCx, PCy):
    a = getdistance1(PAx, PAy, PBx, PBy)
    b = getdistance1(PBx, PBy, PCx, PCy)
    c = getdistance1(PAx, PAy, PCx, PCy)
    if b * b >= c * c + a * a:#钝角返回边
        return c
    if c * c >= a * a + b * b:#钝角返回边
        return b
    else:
        l = (a + b + c) / 2
        if abs(l - a) < 0.1:
            return 0
        else:
            s = math.sqrt(l * (l - a) * (l - b) * (l - c))#海伦公式
            return 2 * s / a #返回高


# 对于一条路段的稀疏部分进行点的添加,获得稠密路段信息
def getDenseWay(initial_result):
    result = initial_result
    for j in range(0, len(result)):
        i = 0
        done = 0
        while done == 0:
            xcor1 = result[j][i * 2 + 2]
            ycor1 = result[j][i * 2 + 3]
            xcor2 = result[j][i * 2]
            ycor2 = result[j][i * 2 + 1]
            a = math.sin(ycor1) * math.sin(ycor2)  # 两点纬度乘积
            b = math.cos(ycor1) * math.cos(ycor2) * math.cos(xcor1 - xcor2)
            dis = math.acos(a + b) * 6371004
            if dis >= 600:
                newxcor = (xcor1 + xcor2) / 2
                newycor = (ycor1 + ycor2) / 2
                result[j].insert(i * 2 + 2, newycor)
                result[j].insert(i * 2 + 2, newxcor)
            else:
                i = i + 1
                roadlength = len(result[j])
                if i == roadlength / 2 - 1:
                    done = 1
    return result


# 使用方法一来计算一下路段中的点的距离到底是多少
def getresult_dis1(result):
    result_dis = []
    for j in range(0, len(result)):
        point_dis = []
        roadlength = len(result[j])
        for i in range(0, int(roadlength / 2 - 1)):
            xcor1 = result[j][i * 2 + 2]
            ycor1 = result[j][i * 2 + 3]
            xcor2 = result[j][i * 2]
            ycor2 = result[j][i * 2 + 1]
            dis = getdistance1(xcor1, ycor1, xcor2, ycor2)
            point_dis.append(dis)
        result_dis.append(point_dis)
    return result_dis


# 使用方法二来计算一下路段中的点的距离到底是多少
def getresult_dis2(result):
    result_dis = []
    for j in range(0, len(result)):
        point_dis = []
        roadlength = len(result[j])
        for i in range(0, int(roadlength / 2 - 1)):
            xcor1 = result[j][i * 2 + 2]
            ycor1 = result[j][i * 2 + 3]
            xcor2 = result[j][i * 2]
            ycor2 = result[j][i * 2 + 1]
            dis = getdistance2(xcor1, ycor1, xcor2, ycor2)
            point_dis.append(dis)
        result_dis.append(point_dis)
    return result_dis

def back_coordinates(PAx, PAy, PBx, PBy, PCx, PCy):
    a = getdistance1(PAx, PAy, PBx, PBy)
    b = getdistance1(PBx, PBy, PCx, PCy)
    c = getdistance1(PAx, PAy, PCx, PCy)
    if b * b >= c * c + a * a:  # 钝角返回边
        return PAx, PAy
    if c * c >= a * a + b * b:  # 钝角返回边
        return PBx, PBy
    else:
        l = (a + b + c) / 2
        if abs(l - a) < 0.1:
            return PCx, PCy
        else:
            s = math.sqrt(l * (l - a) * (l - b) * (l - c))
            p = math.sqrt((c * c - pow(2 * s / a, 2)))
            segement = p / a
            x_symbol = 1
            y_symbol = 1
            if PAx > PBx:
                x_symbol = -1
            if PAy > PBy:
                y_symbol = -1
            middle_x = PAx + x_symbol * (abs(PAx - PBx) * segement)
            middle_y = PAy + y_symbol * (abs(PAy - PBy) * segement)
            # print(2 * s / a, getdistance1(PCx, PCy, middle_x, middle_y))
            return middle_x, middle_y

# 评测这个司机轨迹中的各个点最匹配的路段，返回一个新的轨迹数据序列
# 增加了最匹配路段号，点到路段距离，和距离评测值三列属性
# 输入是司机的轨迹信息traj_frame和路段坐标信息result
def getNewTraj(traj_frame):
    # 定义新增加的三列属性
    coor = pd.read_csv(r'data/split_coordinates_20.csv', low_memory=False, encoding='utf-8')
    # 检索司机的第k个轨迹点
    # meanless = 0
    # all_back = 0
    valid = True
    licence_list = []
    time_list = []
    main_id_list = []
    sub_id_list = []
    modify_x_list = []
    modify_y_list = []
    speed_list = []
    dis_list = []
    score_list = []
    for k in tqdm(range(0, traj_frame.shape[0])):
        point_time = traj_frame.iloc[k, 0]
        licence = traj_frame.iloc[k, 1]
        PCx = traj_frame.iloc[k, 2]
        PCy = traj_frame.iloc[k, 3]
        speed = traj_frame.iloc[k, 4]
        bound = 0.00050
        # pre_min = 100000
        while True:
            x_up_bound = PCx + bound
            x_down_bound = PCx - bound
            y_up_bound = PCy + bound
            y_down_bound = PCy - bound
            new = coor[
                ((coor['A_x'] < x_up_bound) & (coor['A_x'] > x_down_bound) & (
                            (coor['A_y'] < y_up_bound) & (coor['A_y'] > y_down_bound))) | (
                        (coor['B_x'] < x_up_bound) & (coor['B_x'] > x_down_bound) & (
                        (coor['B_y'] < y_up_bound) & (coor['B_y'] > y_down_bound)))]
            if new.empty == True:
                bound = bound * 2
                if bound > 0.004:
                    valid = False
                    break
                continue
            point_point_dis = []
            point_point_value = []
            point_pair = []
            point_id_and_sub_id = []
            for index, item in new.iterrows():
                PAx = item['A_x']
                PAy = item['A_y']
                PBx = item['B_x']
                PBy = item['B_y']
                dis = getPointWayDis(PAx, PAy, PBx, PBy, PCx, PCy)
                value = 1 - math.exp(-dis)
                point_point_dis.append(dis)# 记录计算查询点到轨迹j中第i个线段的实际距离
                point_point_value.append(value)  # 存储计算查询点到轨迹j中第i个线段的评测距离
                point_pair.append([PAx, PAy, PBx, PBy])
                point_id_and_sub_id.append([int(item['main_id']), int(item['sub_id'])])
            check_dis = getdistance1(PCx, PCy, x_up_bound, PCy)
            min_dis = min(point_point_dis)
            if min_dis > check_dis:
                # all_back += 1
                # pre_min = min_dis
                bound = bound * 1.415 #近似根号2
                continue
            else:
                # if pre_min == min_dis:
                #     meanless += 1
                break
        # 寻找与该轨迹点最相近的路径j的编号存入new_frame.iloc[k,0]
        if valid:
            min_dis_index = point_point_dis.index(min_dis)
            road_pair = point_pair[min_dis_index]
            id_pair = point_id_and_sub_id[min_dis_index]
            modify_x, modify_y = back_coordinates(road_pair[0], road_pair[1], road_pair[2], road_pair[3], PCx, PCy)
            modify_x_list.append(modify_x)
            modify_y_list.append(modify_y)
            main_id_list.append(id_pair[0])
            sub_id_list.append(id_pair[1])
            dis_list.append(min_dis)
            score_list.append(min(point_point_value))
        else:
            modify_x_list.append(PCx)
            modify_y_list.append(PCy)
            main_id_list.append(-1)
            sub_id_list.append(-1)
            dis_list.append(-1)
            score_list.append(-1)
        licence_list.append(licence)
        speed_list.append(speed)
        time_list.append(point_time)
    # print(all_back, meanless)
    a = {'licence': licence_list,'main_id':main_id_list, 'sub_id':sub_id_list, 'time': time_list, 'lon': modify_x_list, 'lat': modify_y_list, 'speed':speed_list, 'dis': dis_list, 'score': score_list}
    new_traj = pd.DataFrame(a)
    # 获得新的轨迹数据序列
    return new_traj


# 为dataframe增加新的一列的方法：
# print(data_frame.columns)        #输出表的标题栏
# col_name = data_frame.columns.tolist()
# col_name.insert(col_name.index('序号')+1,'电话号')     #在"序号"列后边插入一个新的列，列名为"电话号"
# data_frame.columns[2] #返回这一列的列名
def form_coordinates_pair():
    # 导入路段数据
    # data_frame = pd.read_excel(cfg.shenzhen_road, sheet_name='Sheet1')
    data_frame = pd.read_csv(cfg.shenzhen_roadDataPathName)
    rownum = data_frame.shape[0]  # data_frame的行数
    colnum = data_frame.shape[1]  # data_frame的列数
    result = list()
    # 取data_frame的位置列属性，清洗并将字符串数字化
    for rowcount in range(0, rownum):
        new = data_frame.iloc[rowcount, 2]
        new = re.sub(' ', ',', new)
        new = new.split(',')
        for i in range(0, len(new)):
            new[i] = float(new[i])
        result.append(new)
    return result

def wash_task(source_path_name, target_path_name):
    # 处理完了道路数据，准备进行司机轨迹的匹配，导入司机轨迹并处理生成新的数据
    traj_frame = pd.read_csv(source_path_name, sep=",")
    # traj_frame.columns = pd.Index(['DriverID', 'Time', 'Xposition', 'Yposition'])
    #traj_frame.shape
    # result 即为路段的经纬度点组成
    logging.info('start:' + source_path_name)
    new_traj = getNewTraj(traj_frame)
    # 将新的司机轨迹数据导出
    # 字段信息：
    new_traj.to_csv(target_path_name, index=False, index_label=None)
    logging.info('finish:' + target_path_name)

if __name__ == '__main__':
    # wash_task(cfg.shenzhen_test, cfg.shenzhen_target)
    print('Parent process %s.' % os.getpid())
    p = Pool(9)
    # p = multiprocessing.Pool(processes=2)
    # 轨迹数据文件夹
    original_files = os.listdir(cfg.shenzhen_fullDataSetPath)
    print("开始处理.......")
    logging.info('start')
    for i in range(0, len(original_files)):#len(original_files)
        # 生成源文件路径名
        filePathName = cfg.shenzhen_fullDataSetPath + '/' + str(i+1) + '.csv'
        targetPathName = cfg.shenzhen_fullDataConcatPath + '/' + str(i+1) + '.csv'
        # 建一个进程
        p.apply_async(wash_task, args=(filePathName, targetPathName))
    p.close()
    p.join()
    logging.info('Well down')
