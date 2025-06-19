from math import sqrt
import random
import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm


from torch.nn import functional as F
from Evaluation import Evaluation 
from utils import load_data

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_absolute_error
import csv


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





class weighted_MSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,inputs,targets,weights):
        return ((inputs - targets)**2 ) * weights
    
def get_feat_labels(df):

    Id_list = list(user_feature.keys())
    df = df[df['user'].isin(Id_list)]

    # 将ratings列转换为标签
    labels = df['rating'].tolist()

    # 将user_index、item_index和labels转换为列表
    user_index = df['user'].tolist()
    item_index = df['item'].tolist()


    user_fea_idx = [user_feature[x] for x in user_index]
    item_fea_idx = [item_feature[x] for x in item_index]

    user_fea_idx = torch.cat(user_fea_idx, dim=0).to(config["device"])
    item_fea_idx = torch.cat(item_fea_idx, dim=0).to(config["device"])

    labels = torch.tensor(labels).to(config["device"])
    labels = labels.float()
    return user_fea_idx, item_fea_idx, labels





def prompt(pretrain_model):
    from model import ColdPrompt
    import pandas as pd


    # 从DataFrame对象创建ui_dict和ur_dict字典
    supp_dict = {user: group for user, group in u_cold_supp.groupby('user')}
    query_dict = {user: group for user, group in u_cold_query.groupby('user')}

    loss_fn = weighted_MSELoss()

    user_list = supp_dict.keys()
    # user_list = random.sample(user_list, 100) 
    meta_path = 3
    prompt_model = ColdPrompt(config['embedding_dim'],meta_path,len(user_list),config).to(config["device"])
    optimizer = torch.optim.Adam(prompt_model.parameters(), lr=config['prompt_lr'], weight_decay=config['prompt_wd']) 
    mae, rmse,ndcg_at_5 = [], [], []
    min_mae,min_rmse,min_ndcg_5 = 10,10,10
    for epoch in range(1,config['prompt_epochs']+1):
        loss = 0
        pretrain_model.eval()
        prompt_embed = pretrain_model.get_prompt_embedding()
        prompt_embed = prompt_embed[:, 0:meta_path, :]
        specific_emb, share_emb = prompt_model(prompt_embed)
        
        for idx, user in enumerate(user_list):
            user_fea_idx, item_fea_idx, labels = get_feat_labels(supp_dict[user])
            with torch.no_grad():
                u_embedding,i_embedding = pretrain_model.get_embed(user_fea_idx,item_fea_idx)
            prompt_embedding = specific_emb[idx]+share_emb
            pred = pretrain_model.regression(u_embedding,i_embedding,prompt_embedding/2)
            loss += F.mse_loss(pred, labels)
        # loss = loss/len(user_list)

        
        print(epoch,loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch %5 == 0:
            with torch.no_grad():       
                prompt_embed = pretrain_model.get_prompt_embedding()
                prompt_embed = prompt_embed[:, 0:meta_path, :]
                specific_emb, share_emb = prompt_model(prompt_embed)
                for idx, user in enumerate(user_list):
                    user_fea_idx, item_fea_idx, labels = get_feat_labels(query_dict[user])
                    with torch.no_grad():
                        u_embedding,i_embedding = pretrain_model.get_embed(user_fea_idx,item_fea_idx)

                    prompt_embedding = specific_emb[idx]+share_emb
                    pred = pretrain_model.regression(u_embedding,i_embedding,prompt_embedding/2)
                    _mae,_rmse,_ndcg_5 = evaluate(pred,labels)
                    # print(_mae,_rmse,_ndcg_5)
                    mae.append(_mae)
                    rmse.append(_rmse)
                    ndcg_at_5.append(_ndcg_5)
                print("-------------------------mean-------------------------")
                # print(_mae,_rmse,_ndcg_5 )
                print('user: {:d} mae: {:.5f}, rmse: {:.5f}, ndcg@5: {:.5f}, loss: {:.5f}'.
                        format(user,np.mean(mae), np.mean(rmse), np.mean(ndcg_at_5),loss.item()))
            min_mae,min_rmse,min_ndcg_5 = min(min_mae,np.mean(mae)),min(min_rmse,np.mean(rmse)),min(min_ndcg_5,np.mean(ndcg_at_5))

    res = {'loss': loss.item(), 'mae':min_mae,'rmse':min_rmse,'ndcg@5':min_ndcg_5}
    config.update(res)
        
        # 将结果转化为DataFrame，并保存到CSV文件中

    with open("./results/"+data_set+"/prompt/prompt_"+pred_type+".csv", 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=config.keys())
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(config)
    return min_mae, np.mean(rmse), np.mean(ndcg_at_5)


def prompt2(pretrain_model):
    from model import ColdPrompt,ColdPrompt2
    import pandas as pd


    # 从DataFrame对象创建ui_dict和ur_dict字典
    supp_dict = {user: group for user, group in u_cold_supp.groupby('user')}
    query_dict = {user: group for user, group in u_cold_query.groupby('user')}

    loss_fn = weighted_MSELoss()

    user_list = supp_dict.keys()
    # user_list = random.sample(user_list, 100) 
    
    mae, rmse,ndcg_at_5 = [], [], []
    with tqdm(total=len(user_list), desc='(T)') as pbar:
        for user in user_list:
            user_fea_idx, item_fea_idx, labels = get_feat_labels(supp_dict[user])
            prompt_model = ColdPrompt2(config['embedding_dim'],config["num_meta_paths"],config['prompt_type'],config).to(config["device"])
            optimizer = torch.optim.Adam(prompt_model.parameters(), lr=config['prompt_lr'], weight_decay=config['prompt_wd']) 
            with torch.no_grad():
                u_embedding,i_embedding = pretrain_model.get_embed(user_fea_idx,item_fea_idx)
            
            patience = 5
            counter = 0
            min_loss = 100000
            
            for epoch in range(config['prompt_epochs']):
                pretrain_model.train()

                prompt_embed = pretrain_model.get_prompt_embedding()
                prompt_embed = prompt_model(prompt_embed)
                # ui_embedding = torch.cat([u_embedding,i_embedding], dim=1)

                # uim_embedding = ui_embedding*prompt_embed
                # prompt_embed = prompt_embed.expand(u_embedding.shape[0],-1)
                pred = pretrain_model.regression(u_embedding,i_embedding,prompt_embed)
                # loss = F.mse_loss(pred, labels)
                loss_weight = torch.ones_like(labels)
                # loss_weight = (pred-labels)**2
                loss = loss_fn(pred,labels,loss_weight).mean(dim = -1)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                

                if min_loss > loss.item():
                    min_loss = loss.item()
                    counter = 0
                    torch.save(prompt_model.state_dict(), "./save_model/"+data_set+"ColdUserPrompt"+pred_type+".pkl")
                else:
                    counter += 1
                    if counter >= patience:
                        # print('Early stopping after epoch {}!'.format(epoch + 1))
                        break
                    
            
            prompt_model_state_dict = torch.load("./save_model/"+data_set+"ColdUserPrompt"+pred_type+".pkl")
            prompt_model.load_state_dict(prompt_model_state_dict)
            # print(user,loss.item(),labels.detach().cpu().numpy(),pred.detach().cpu().numpy(),epoch)


            user_fea_idx, item_fea_idx, labels = get_feat_labels(query_dict[user])
            with torch.no_grad():
                pretrain_model.eval()
                u_embedding,i_embedding = pretrain_model.get_embed(user_fea_idx,item_fea_idx)

                prompt_embed = pretrain_model.get_prompt_embedding()
                prompt_embed = prompt_model(prompt_embed)
                # ui_embedding = torch.cat([u_embedding,i_embedding], dim=1)

                # uim_embedding = ui_embedding*prompt_embed
                # prompt_embed = prompt_embed.expand(u_embedding.shape[0],-1)
                pred = pretrain_model.regression(u_embedding,i_embedding,prompt_embed)
                _mae,_rmse,_ndcg_5 = evaluate(pred,labels)
                # print(_mae,_rmse,_ndcg_5)
                mae.append(_mae)
                rmse.append(_rmse)
                ndcg_at_5.append(_ndcg_5)
            # user_result = {'user':user,'mae':_mae,'rmse':_rmse,'ndcg@5':_ndcg_5,'loss':loss.item()}
            # import csv
            # with open("./results/"+data_set+"/prompt/user_loss.csv", 'a', newline='') as file:
            #     writer = csv.DictWriter(file, fieldnames=user_result.keys())
            #     if file.tell() == 0:
            #         writer.writeheader()
            #     writer.writerow(user_result)


            print("-------------------------mean-------------------------")
            # print(_mae,_rmse,_ndcg_5 )
            print('user: {:d} mae: {:.5f}, rmse: {:.5f}, ndcg@5: {:.5f}, loss: {:.5f}'.
                    format(user,np.mean(mae), np.mean(rmse), np.mean(ndcg_at_5),loss.item()))
            pbar.update()


    res = {'loss': loss.item(), 'mae':np.mean(mae),'rmse':np.mean(rmse),'ndcg@5':np.mean(ndcg_at_5)}
    config.update(res)
        
        # 将结果转化为DataFrame，并保存到CSV文件中

    with open("./results/"+data_set+"/prompt/prompt_"+pred_type+".csv", 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=config.keys())
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(config)
    return np.mean(mae), np.mean(rmse), np.mean(ndcg_at_5)


def train_miniBatch(model):
    loss_fcn = weighted_MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["wd"]
    )
    

    load_model = False
    



    if not load_model:
        
        patience = 500
        counter = 0
        min_loss = 100000
        with tqdm(total=config["epochs"], desc='(T)') as pbar:
            for epoch in range(config["epochs"]):
                loss = 0
                model.train()
                loss_list = []
                for mp_idx in range(config["num_meta_paths"]):
                    user_fea_idx, item_fea_idx, labels = get_feat_labels(train_list[mp_idx].sample(config['batch_size']))
                    pred = model.train_encoder(user_fea_idx, item_fea_idx,mp_idx)
                    loss_list.append(F.mse_loss(labels,pred))
                loss_list = torch.stack(loss_list, dim=0)
                att = F.softmax(loss_list, dim=0)
                # att = torch.tensor([0.5]).to(config["device"])
                loss = torch.sum(att*loss_list)
                # loss = torch.mean(loss_list)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print("epoch:",epoch,"loss:",  loss.item(),"loss_list: ",loss_list.detach().cpu().numpy())

                if min_loss > loss.item():
                    min_loss = loss.item()
                    counter = 0
                    torch.save(model.state_dict(), "./save_model/MetaPathPrompt_"+pred_type+".pkl")
                else:
                    counter += 1
                    if counter >= patience:
                        print('Early stopping after epoch {}!'.format(epoch + 1))
                        break
                

                pbar.update()
                    
    trained_state_dict = torch.load("./save_model/MetaPathPrompt_"+pred_type+".pkl")
    model.load_state_dict(trained_state_dict)

    
    return 




def evaluate(pred,labels):
    pred = pred.cpu().numpy()
    labels = labels.cpu().numpy()
    import numpy as np
    from sklearn.metrics import ndcg_score,mean_absolute_error,mean_squared_error
    mse = mean_squared_error(labels, pred)
    mae = mean_absolute_error(labels, pred)
    rmse=np.sqrt(mse)
    ndcg_5 = ndcg_score(labels.reshape(1, labels.shape[0]), pred.reshape(1, pred.shape[0]), k=3)
    return mae, rmse, ndcg_5




if __name__ == "__main__":

    data_set = "dbook"

    train_list,u_cold_supp,u_cold_query, user_feature, item_feature = load_data(data_set)
    # train_list[0],train_list[1]=train_list[1],train_list[0]
    
    
    if data_set == 'movielens':
        from Config import config_ml as config
    elif data_set == 'yelp':
        from Config import config_yelp as config
    elif data_set == 'dbook':
        from Config import config_db as config

    from model import MetaPathPrompt

        

    print(config)
    pred_type = "main"
    model = MetaPathPrompt(config=config, device=config["device"])
    
    # train_metaPathPrompt(model)
    train_miniBatch(model)

    print("------------------------prompt----------------------------")
    prompt(model)




