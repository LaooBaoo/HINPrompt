# coding: utf-8
# author: lu yf
# create date: 2019-11-27 13:14
import math
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


class Evaluation:
    def __init__(self):
        self.k = 5

    def prediction(self, real_score, pred_score):
        MAE = mean_absolute_error(real_score, pred_score)
        RMSE = math.sqrt(mean_squared_error(real_score, pred_score))
        return MAE, RMSE

    def dcg_at_k(self,scores):
        # assert scores
        return scores[0] + sum(sc / math.log(ind+1, 2) for sc, ind in zip(scores[1:], range(2, len(scores) + 1)))

    def ndcg_at_k(self, real_scores, predicted_scores):
        idcg = self.dcg_at_k(sorted(real_scores, reverse=True))
        return (self.dcg_at_k(predicted_scores) / idcg) if idcg > 0.0 else 0.0

    def ranking(self, real_score, pred_score, k):
        # ndcg@k
        sorted_idx = sorted(np.argsort(real_score)[::-1][:k])  # get the index of the top k real score
        r_s_at_k = real_score[sorted_idx]
        p_s_at_k = pred_score[sorted_idx]

        ndcg_5 = self.ndcg_at_k(r_s_at_k, p_s_at_k)
        #
        # ndcg = {}
        # for k in k_list:
        #     sorted_idx = sorted(np.argsort(real_score)[::-1][:k])
        #     r_s_at_k = real_score[sorted_idx]
        #     p_s_at_k = pred_score[sorted_idx]
        #
        #     ndcg[k] = self.ndcg_at_k(r_s_at_k, p_s_at_k)
        return ndcg_5
    
    def NDCG_at_k(self,real_score, pred_score, k):
        # 计算DCG
        sorted_idx = np.argsort(pred_score)[::-1]
        real_score_sorted = real_score[sorted_idx][:k]
        dcg = np.sum((2 ** real_score_sorted - 1) / np.log2(np.arange(2, k+2)))
        print(real_score_sorted)
        # 计算IDCG
        real_score_sorted_desc = np.sort(real_score)[::-1][:k]
        idcg = np.sum((2 ** real_score_sorted_desc - 1) / np.log2(np.arange(2, k+2)))

        # 计算NDCG
        ndcg = dcg / idcg if idcg > 0 else 0
        return ndcg







