
import pickle
import random
from pprint import pprint
import pandas as pd
import dgl

import numpy as np
import torch
from dgl import load_graphs
from dgl.data.utils import _get_dgl_url, download, get_download_dir
from scipy import io as sio, sparse

def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def get_feat_labels(user_feature, item_feature, df):

    Id_list = list(user_feature.keys())
    df = df[df['user'].isin(Id_list)]

    # 将ratings列转换为标签
    labels = df['rating'].tolist()

    # 将user_index、item_index和labels转换为列表
    user_index = df['user'].tolist()
    item_index = df['item'].tolist()


    user_fea_idx = [user_feature[x] for x in user_index]
    item_fea_idx = [item_feature[x] for x in item_index]

    user_fea_idx = torch.cat(user_fea_idx, dim=0)
    item_fea_idx = torch.cat(item_fea_idx, dim=0)

    return user_fea_idx, item_fea_idx, labels

def random_sampling(df, k):
    supp_data = []
    query_data = []
    grouped = df.groupby('user')
    
    for _, group in grouped:
        if len(group)<=k+1:
            continue
        selected_indices, unselected_indices = select_indices(len(group),k)
        supp_data.append(group.iloc[selected_indices])
        query_data.append(group.iloc[unselected_indices])

    
    return pd.concat(supp_data), pd.concat(query_data)

def random_sampling_normal(test_df, k,train_df):
    train_data = []
    supp_data = []
    query_data = []
    grouped = test_df.groupby('user')
    
    for _, group in grouped:
        if len(group)<=k+1:
            continue
        selected_indices, unselected_indices = select_indices(len(group),k)
        if group['user'].iloc[0] in train_df['user'].tolist() :
            train_data.append(train_df[train_df['user']==group['user'].iloc[0]])
            query_data.append(group)
        else:
            supp_data.append(group.iloc[selected_indices])
            query_data.append(group.iloc[unselected_indices])
    train_data_df = pd.concat(train_data)
    if len(supp_data) >0:
        supp_data = pd.concat(supp_data)
        train_data_df = pd.concat([supp_data,train_data_df])

    return train_data_df, pd.concat(query_data)

def select_indices(N, k):
    indices = np.arange(N)
    selected_indices = np.random.choice(indices, size=k, replace=False)
    unselected_indices = np.setdiff1d(indices, selected_indices)
    return selected_indices, unselected_indices

def load_dbook():
    metapath = ["new_u_b_df.csv","new_u_b_a_b_df.csv","new_u_b_u_b_df.csv"]
    train_list= []

    with open("./data/dbook/user_feature.pkl", "rb") as f:
        user_feature = pickle.load(f)

    with open("./data/dbook/item_feature.pkl", "rb") as f:
        item_feature = pickle.load(f)

    for path in metapath:
        df = pd.read_csv("./data/dbook/"+path, delimiter=',')
        df = df.rename(columns={'user1': 'user'})
        df = df.rename(columns={'user2': 'user'})
        df = df.rename(columns={'book': 'item'})


        df = df[df['user'].isin(list(user_feature.keys()))]
        df = df[df['item'].isin(list(item_feature.keys()))]
        
        train_list.append(df)
    interaction_num = 10
    state = 'normal'
    cold_data = pd.read_csv("./data/dbook/new_"+state+"_df.csv", delimiter=',')
    cold_data = cold_data[cold_data['user'].isin(list(user_feature.keys()))]
    cold_data = cold_data[cold_data['item'].isin(list(item_feature.keys()))]
    if state != 'normal':
        cold_supp,cold_query = random_sampling(cold_data, interaction_num)
    else:
        cold_supp,cold_query = random_sampling_normal(cold_data, interaction_num,train_list[0])

    cold_supp = cold_supp.rename(columns={'user1': 'user'})
    cold_supp = cold_supp.rename(columns={'user2': 'user'})
    cold_supp = cold_supp.rename(columns={'book': 'item'})
    cold_query = cold_query.rename(columns={'user1': 'user'})
    cold_query = cold_query.rename(columns={'user2': 'user'})
    cold_query = cold_query.rename(columns={'book': 'item'})

    print(cold_supp.shape, cold_query.shape,interaction_num,state)
    return train_list,cold_supp,cold_query, user_feature, item_feature


def load_data(dataset):
    if dataset == "dbook":
        return load_dbook()
    else:
        return NotImplementedError("Unsupported dataset {}".format(dataset))

