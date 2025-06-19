import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GATConv
import dgl
import torch.nn.init as init
from torch.autograd import Variable

class elementwise_prompt(nn.Module):
    def __init__(self,input_dim):
        super(elementwise_prompt, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,input_dim))
        # self.max_n_num=input_dim
        # self.reset_parameters()
        # self.weight.data.fill_(1)
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, graph_embedding):
        graph_embedding=graph_embedding*self.weight
        return graph_embedding



class ColdPrompt(torch.nn.Module):
    def __init__(self,embedding_dim,num_meta_paths,user_count,config):
        super(ColdPrompt, self).__init__()
        self.embedding_dim=embedding_dim*3
        self.num_meta_paths = num_meta_paths
        self.share_prompt = nn.Linear(self.embedding_dim*num_meta_paths, self.embedding_dim)
        self.specific_prompt = nn.ModuleList()
        self.user_count = user_count
        for i in range(user_count):
            self.specific_prompt.append(nn.Linear(self.embedding_dim*num_meta_paths, self.embedding_dim))


    


    def forward(self, weight):

        prompt_emb = weight.reshape(weight.shape[0], weight.shape[1] * weight.shape[2])
        share_emb = (self.share_prompt(prompt_emb)+torch.mean(weight,dim=1))
        task_specific = None
        for i in range(self.user_count):
            if task_specific is None:
                task_specific = self.specific_prompt[i](prompt_emb)
            else:
                task_specific = torch.cat([task_specific, self.specific_prompt[i](prompt_emb)],dim=0)
        
        return task_specific, torch.mean(weight,dim=1)



    

class AttentionPrompt(nn.Module):
    def __init__(self,num_meta_path,config):
        super(AttentionPrompt,self).__init__()
        self.head = 4
        self.attention = nn.ParameterList()
        for i in range(self.head):
            self.attention.append(torch.nn.Parameter(torch.Tensor(1,num_meta_path)).to(config['device']))
            torch.nn.init.xavier_uniform_(self.attention[i])
            
        
        
    def forward(self, metapath_prompt):
        N,p, d =  metapath_prompt.shape[0], metapath_prompt.shape[1], metapath_prompt.shape[2]
        # 对权重进行归一化，使其和为1
        prompt_list = []
        for i in range(self.head):
            weight = self.attention[i]
            normalized_weights = weight / torch.sum(weight)
        
            # 利用归一化后的权重对矩阵进行加权求和
            prompt = torch.mm(normalized_weights,  metapath_prompt.squeeze())
            prompt_list.append(prompt)
        prompt_list = torch.stack(prompt_list, dim=1)
        prompt_list = torch.mean(prompt_list,dim=1)
        return prompt_list


class ElementPrototypesRegression(nn.Module):
    def __init__(self,user_dim,item_dim,hid_dim,proto_dim,layer_num,device):
        super(ElementPrototypesRegression, self).__init__()
        self.LABEL_MIN = 1
        self.LABEL_MAX = 5
        self.device = device
        self.prototypes = F.normalize(torch.randn(1,proto_dim), p=2).to(self.device)

        self.type = "skip"
        self.head_num = 4
        self.layer_num = 10

        self.bn = nn.BatchNorm1d(proto_dim)
        self.use_bn = False
        
        if self.type=="skip":
            self.uimModule = nn.Sequential(nn.Linear(user_dim+item_dim, hid_dim),nn.ReLU(),nn.Linear(hid_dim, user_dim+item_dim)).to(self.device)
            self.linear = nn.Linear(user_dim+item_dim, proto_dim).to(self.device)
        elif self.type=="mlp":
            self.uimModule = nn.Sequential()
            self.uimModule.add_module('linear1', nn.Linear(user_dim+item_dim, hid_dim))
            self.uimModule.add_module('relu1', nn.ReLU())
            for i in range(2, self.layer_num):
                self.uimModule.add_module('linear'+str(i), nn.Linear(hid_dim, hid_dim))
                self.uimModule.add_module('relu'+str(i), nn.ReLU())
            self.uimModule.add_module('linear'+str(self.layer_num), nn.Linear(hid_dim, proto_dim))
        elif self.type == "multi_head":
            self.uimModule = nn.ModuleList()
            for i in range(self.head_num):
                self.uimModule.append(nn.Sequential(nn.Linear(user_dim+item_dim, hid_dim),nn.ReLU(),nn.Linear(hid_dim, proto_dim)))
            self.transform = nn.Linear(self.head_num*proto_dim, proto_dim)


            

    def forward(self, uim_embed):
        if self.type == "skip":
            uim_embed = self.uimModule(uim_embed)+uim_embed
            uim_embed = self.linear(uim_embed)
        elif self.type == "mlp":
            uim_embed = self.uimModule(uim_embed)
        elif self.type == "multi_head":
            uim_embed = torch.cat([self.uimModule[i](uim_embed) for i in range(self.head_num)],dim=1)
            uim_embed = self.transform(uim_embed)

        # uim_embed = F.normalize(uim_embed, p=2,dim=1)
        pred = torch.mm(uim_embed, self.prototypes.t())
        pred = ((pred+1)/2)*(self.LABEL_MAX-self.LABEL_MIN)+self.LABEL_MIN
        return pred.squeeze(1)
    





class PredictionLayer(nn.Module):
    def __init__(self,user_dim,item_dim,prompt_dim,hid_dim,device):
        super(PredictionLayer, self).__init__()
        self.device = device
        dropout_rate = 0
        self.user_attention = nn.ModuleList([nn.Linear(32, 128),nn.Linear(32, 128),nn.Linear(32, 128),nn.Linear(32, 128)])
        self.item_attention = nn.ModuleList([nn.Linear(32, 128),nn.Linear(32, 128)])
        self.uimModule = nn.Sequential(
            nn.Linear(user_dim+item_dim, hid_dim,bias=True),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hid_dim, hid_dim,bias=True),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hid_dim, hid_dim,bias=True),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hid_dim, 1,bias=True)

        ).to(self.device)

       






    def forward(self, user_emb, item_emb, prompt_embed):

        ui_embedding = torch.cat([user_emb,item_emb], dim=1)
        uim_embedding = prompt_embed*ui_embedding
        # uim_embedding = ui_embedding


        pred = self.uimModule(uim_embedding)+1
        # user_embedding = self.user_layer(user_emb)
        # item_embedding = self.item_layer(item_emb)
        # pred = user_embedding*item_embedding
        # print(pred.shape)
        # exit()

        return pred.squeeze(1)
        
        

    

class MetaPathPrompt(nn.Module):
    def __init__(self, config, device):
        super(MetaPathPrompt, self).__init__()
        self.device = device
        self.config = config
        
        self.device = device
        if self.config['dataset'] == 'movielens':
            from EmbeddingInitializer import UserEmbeddingML, ItemEmbeddingML, PromptEmbedding
            self.item_emb = ItemEmbeddingML(self.config)
            self.user_emb = UserEmbeddingML(self.config)
            self.prompt_emb = PromptEmbedding(self.config)
            self.user_fea_num = 4
            self.item_fea_num = 2
        elif self.config['dataset'] == 'yelp':
            from EmbeddingInitializer import UserEmbeddingYelp, ItemEmbeddingYelp, PromptEmbedding
            self.item_emb = ItemEmbeddingYelp(self.config)
            self.user_emb = UserEmbeddingYelp(self.config)
            self.prompt_emb = PromptEmbedding(self.config)
            self.user_fea_num = 1
            self.item_fea_num = 2
        elif self.config['dataset'] == 'dbook':
            from EmbeddingInitializer import UserEmbeddingDB, ItemEmbeddingDB
            self.item_emb = ItemEmbeddingDB(self.config)
            self.user_emb = UserEmbeddingDB(self.config)
            self.user_fea_num = 1
            self.item_fea_num = 1
        
        self.user_emb = self.user_emb.to(self.device)
        self.item_emb = self.item_emb.to(self.device)
        self.prompt_emb = self.prompt_emb.to(self.device)
        
        # self.share_prompt = nn.Linear(self.config['prompt_dim'],self.config['prompt_dim']).to(self.device)
        reg_type = "mlp"
        self.meta_path_prompt = []


        # self.temp = self.prompt_embedding[0].clone()

        # for i in range(config["num_meta_paths"]):
        #     self.meta_path_prompt.append(elementwise_prompt(config['embedding_dim']*(self.user_fea_num+self.item_fea_num)).to(self.device))
        if reg_type == "proto":
            self.regression = ElementPrototypesRegression(config['embedding_dim']*self.user_fea_num,config['embedding_dim']*self.item_fea_num,config['hid_dim'], config['proto_dim'],config['layer_num'],device=self.device).to(self.device)
        elif reg_type == "mlp":
            self.regression = PredictionLayer(config['embedding_dim']*self.user_fea_num,config['embedding_dim']*self.item_fea_num,config['prompt_dim'],config['hid_dim'],device=self.device).to(self.device)


    def get_embed(self,user_fea_idx,item_fea_idx):


        u_embedding = self.user_emb(user_fea_idx)
        i_embedding = self.item_emb(item_fea_idx)

    
        return u_embedding, i_embedding

    def reset_parameters(self):
        for i in range(self.num_meta_paths):
            torch.nn.init.xavier_uniform_(self.meta_path_emb[i])


    def get_prompt_embedding(self):
        prompt_weight = []

        for i in range(self.config["num_meta_paths"]):
            prompt_embedding = self.prompt_emb(torch.tensor([i]).to(self.device))
            # prompt_embedding = self.share_prompt(prompt_embedding)
            prompt_weight.append(prompt_embedding)
            
        prompt_weight = torch.stack(prompt_weight, dim=1)
        return prompt_weight
    

    def train_encoder(self,user_fea_idx,item_fea_idx,mp_idx):
        u_embedding, i_embedding = self.get_embed(user_fea_idx,item_fea_idx)

        idx = torch.tensor([mp_idx]).repeat(u_embedding.shape[0]).to(self.device)
        prompt_embedding = self.prompt_emb(idx)
        # prompt_embedding = self.share_prompt(prompt_embedding)
        # ui_embedding = torch.cat([u_embedding,i_embedding], dim=1)

        # uim_embedding = prompt_embedding*ui_embedding
        pred = self.regression(u_embedding, i_embedding,prompt_embedding)
        return pred
        

    
    def forward(self,user_fea_idx,item_fea_idx):
        u_embedding, i_embedding = self.get_embed(user_fea_idx,item_fea_idx)
        meta_path_emb = self.get_prompt_embedding()
        
        prompt_embedding = torch.mean(meta_path_emb,dim=1)
        # prompt_embedding = meta_path_emb[:,:1,:].squeeze(1)
        # prompt_embedding = self.share_prompt(prompt_embedding)
        
        # ui_embedding = torch.cat([u_embedding,i_embedding], dim=1)
        # uim_embedding = prompt_embedding*ui_embedding
        # pred = self.regression(uim_embedding)

        pred = self.regression(u_embedding, i_embedding,prompt_embedding)
        return pred

