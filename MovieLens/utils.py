
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

def random_sampling(df, k,train_df):
    train_data = []
    supp_data = []
    query_data = []
    grouped = df.groupby('user')
    
    for _, group in grouped:
        if len(group)<k+5:
            continue
        selected_indices, unselected_indices = select_indices(len(group),k)  
        supp_data.append(group.iloc[selected_indices])
        query_data.append(group.iloc[unselected_indices])
    supp_data = pd.concat(supp_data)

    return supp_data, pd.concat(query_data)

def select_indices(N, k):
    indices = np.arange(N)
    selected_indices = np.random.choice(indices, size=k, replace=False)
    unselected_indices = np.setdiff1d(indices, selected_indices)
    return selected_indices, unselected_indices

def load_movielens():
    metapath = ["new_u_m_df.csv","new_u_m_a_m_df_more.csv","new_u_m_d_m_df_more.csv","new_u_m_u_m_df.csv"]
    train_list= []

    with open("./data/movielens/user_feature.pkl", "rb") as f:
        user_feature = pickle.load(f)

    with open("./data/movielens/item_feature.pkl", "rb") as f:
        item_feature = pickle.load(f)

    for path in metapath:
        df = pd.read_csv("./data/movielens/"+path, delimiter=',')
        df = df.rename(columns={'user1': 'user'})
        df = df.rename(columns={'user2': 'user'})
        df = df.rename(columns={'movie': 'item'})


        df = df[df['user'].isin(list(user_feature.keys()))]
        df = df[df['item'].isin(list(item_feature.keys()))]
        
        train_list.append(df)

    cold_data = pd.read_csv("./data/movielens/new_user_and_item_cold_df.csv", delimiter=',')
    # cold_supp = pd.read_csv("./data/movielens/user_and_item_cold_supp.csv", delimiter=',')
    # cold_query = pd.read_csv("./data/movielens/user_and_item_cold_query.csv", delimiter=',')

    # cold_supp = pd.read_csv("./data/movielens/item_cold_supp.csv", delimiter=',')
    # cold_query = pd.read_csv("./data/movielens/item_cold_query.csv", delimiter=',')

    # cold_supp = pd.read_csv("./data/movielens/user_and_item_cold_supp.csv", delimiter=',')
    # cold_query = pd.read_csv("./data/movielens/user_and_item_cold_query.csv", delimiter=',')

    # cold_supp = pd.read_csv("./data/movielens/warm_up_supp_30.csv", delimiter=',')
    # cold_query = pd.read_csv("./data/movielens/warm_up_query_30.csv", delimiter=',')


    # # 调用函数进行抽样
    # cold_supp = random_sampling(cold_data, k=5)
    # print(cold_supp)
   
    cold_supp,cold_query = random_sampling(cold_data, 10,train_list[0])
    cold_supp = cold_supp.rename(columns={'user1': 'user'})
    cold_supp = cold_supp.rename(columns={'user2': 'user'})
    cold_supp = cold_supp.rename(columns={'movie': 'item'})
    cold_query = cold_query.rename(columns={'user1': 'user'})
    cold_query = cold_query.rename(columns={'user2': 'user'})
    cold_query = cold_query.rename(columns={'movie': 'item'})

    print(cold_supp.shape, cold_query.shape)
    


    return train_list,cold_supp,cold_query, user_feature, item_feature




def load_data(dataset):
    if dataset == "movielens":
        return load_movielens()
    else:
        return NotImplementedError("Unsupported dataset {}".format(dataset))

