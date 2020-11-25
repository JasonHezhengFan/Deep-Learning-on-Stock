import pandas as pd
import numpy as np
import time
from datetime import datetime

'''
将数据展开成时间序列
输入: 
    join_df: [X, y] 的 dataframe
    gap_day: 前后时间超过 gap_day 就将原 dataframe 切成两段
    roll_day: 展开成时间序列的长度
返回: 处理成序列的 Index, Data, Label 的 list
'''

# def __debug__check_index():  

def unroll_data(join_df, gap_day, roll_day, debug=False):
    # 把前后时间差距大的地方切开
    start = time.time()

    all_dfs = []

    for code, grp in join_df.groupby('stock_code'):
        grp = grp.sort_index()
        split_point = [0]
        for i in range(len(grp.index) - 1):
            if type(grp.index[i][1]) == str:
                current_day = datetime.strptime(grp.index[i+1][1], '%Y-%m-%d')
                pre_day = datetime.strptime(grp.index[i][1], '%Y-%m-%d')
                gap = (current_day - pre_day).days
            else:
                gap = grp.index[i+1][1] - grp.index[i][1]
            if gap > gap_day:
                if debug:
                    print(grp.index[i+1][1])
                    print(grp.index[i][1])
                    print(gap)
                split_point.append(i+1)
        split_point.append(len(grp.index))

        for i in range(len(split_point) - 1):
            if split_point[i+1] - split_point[i] > roll_day:
                all_dfs.append(grp.iloc[split_point[i]:split_point[i+1]])
    time_spend = time.time() - start
    print('Filter large gap...\nTime: %.3f'%(time_spend))
    
    # 把数据展开成sequence
    data_list = []
    label_list = []
    index_list = []

    start = time.time()

    for df in all_dfs:
        grp_data = df.sort_index()
        data_fetch = []
        label_fetch = []
        index = []

        for i in range(grp_data.shape[0] - roll_day):
            data_fetch.append(list(range(i, i + roll_day)))
            label_fetch.append(i + roll_day - 1)
            index.append(grp_data.index[i + roll_day - 1])        

        data_fetch = np.array(data_fetch)
        label_fetch = np.array(label_fetch)

        data = np.take(grp_data.values[:, :-1], data_fetch, axis=0)
        label = np.take(grp_data.values[:,-1], label_fetch, axis=0)

        index_list = index_list + index
        data_list.append(data)
        label_list.append(label)

    data_list = np.concatenate(data_list, axis=0)
    label_list = np.concatenate(label_list, axis=0)

    print('Unrolling Data...\nTime: %.3f s'%(time.time() - start))
    
    return index_list, data_list, label_list