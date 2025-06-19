# coding: utf-8
# author: lu yf
# create date: 2019-11-20 19:46

config_db = {
    'dataset': 'dbook',
    'mp': ['ub'],
    # 'mp': ['ub','ubab'],
    # 'mp': ['ub','ubab','ubub'],
    # 'mp': ['ub','uub','ubab'],
    'use_cuda': True,
    'file_num': 10,  # each task contains 10 files

    # user
    'num_location': 453,
    'num_fea_item': 2,

    # item
    'num_publisher': 1698,
    'num_fea_user': 1,
    'item_fea_len': 1,


	# 'num_epochs': 500,
    # 'lr': 0.09,
    # 'weight_decay':0.066,
    # 'embedding_dim': 16,
    # 'mlp_hid_dim':64,

    'epochs': 50,
    'lr': 0.002,
    'wd':0.0001,
    'embedding_dim': 32,
    'hid_dim':256,
    'num_meta_paths':3,
    'layer_num': 3,

    'proto_dim':16,
    'prompt_lr': 0.001,
    'prompt_wd': 0.05,
    'prompt_epochs': 10,
    'prompt_dim':64,
    'prompt_type': 'elementwise',
    'batch_size': 65536,
    'device': "cuda:4",
}


config_ml = {
    'dataset': 'movielens',
    # 'mp': ['um'],
    # 'mp': ['um','umdm'],
    # 'mp': ['um','umam','umdm'],
    'mp': ['um','umum','umam','umdm'],
    'use_cuda': True,
    'file_num': 12,  # each task contains 12 files for movielens

    # item
    'num_rate': 6,
    'num_genre': 25,
    'num_fea_item': 2,
    'item_fea_len': 26,

    # user
    'num_gender': 2,
    'num_age': 7,
    'num_occupation': 21,
    'num_zipcode': 3402,
    'num_fea_user': 4,

    'epochs': 100,
    'lr': 0.002,
    'wd':0.001,
    'embedding_dim': 8,
    'hid_dim':64,
    'num_meta_paths':3,
    'layer_num': 3,

    'proto_dim':16,
    'prompt_lr': 0.001,
    'prompt_wd': 0.001,
    'prompt_epochs': 20,
    'prompt_dim':128,
    'prompt_type': 'elementwise',
    'batch_size': 32768,
    'device': "cuda:7",
}


config_yelp = {
    'dataset': 'yelp',
    # 'mp': ['ubub'],
    'mp': ['ub','ubcb','ubtb','ubub'],
    'use_cuda': True,
    'file_num': 12,  # each task contains 12 files

    # item
    'num_stars': 9,
    'num_postalcode': 6127,
    'num_fea_item': 2,
    'item_fea_len': 2,

    # user
    'num_fans': 412,
    'num_avgrating': 359,
    'num_fea_user': 2,

    # model setting
    'embedding_dim': 32,
    'user_embedding_dim': 32*2,  # 1 features
    'item_embedding_dim': 32*2,  # 1 features

    'first_fc_hidden_dim': 64,
    'second_fc_hidden_dim': 64,

    'mp_update': 1,
    'local_update': 1,
    'lr': 5e-4,
    'mp_lr': 1e-3,
    'local_lr': 1e-3,
    'batch_size': 32,  # for each batch, the number of tasks
    'num_epoch': 50,
    'neigh_agg': 'mean',
    # 'neigh_agg': 'attention',
    'mp_agg': 'mean',
    # 'mp_agg': 'attention',
}


states = ["meta_training","warm_up", "user_cold_testing", "item_cold_testing", "user_and_item_cold_testing"]
