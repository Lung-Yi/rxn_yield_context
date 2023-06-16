# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 11:48:13 2021

@author: Lung-Yi

Loss function for listwise ranking training.
"""
import torch
import torch.nn.functional as F
from itertools import permutations
PADDED_Y_VALUE = -1
DEFAULT_EPS = 1e-10
DEFAULT_TOPK = 2

def listNet_top_one(y_pred, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE):
    """
    ListNet loss introduced in "Learning to Rank: From Pairwise Approach to Listwise Approach".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    mask = y_true == padded_value_indicator
    y_pred[mask] = float('-inf')
    y_true[mask] = float('-inf')

    preds_smax = F.softmax(y_pred, dim=1)
    true_smax = F.softmax(y_true, dim=1)

    preds_smax = preds_smax + eps
    preds_log = torch.log(preds_smax)
    
    return F.kl_div(preds_log, true_smax, reduction='none').sum(1).mean()
    # return torch.mean(-torch.sum(true_smax * preds_log, dim=1))

def compute_prob(A=torch.Tensor([3,2,0]), index = [0,1]):
    index = list(index)
    smax = F.softmax(A, dim=0)
    prob = smax[index[0]].clone()
    for i, idx in enumerate(index[1:]):
        prob *= smax[idx]/( 1 - torch.sum(smax[index[0:i+1]]) )
    return prob

def calculate_pair_kl_div(y_pred, y_true, mask, topk_for_batch, eps):
    comb = [i for i, m in enumerate(mask) if not m]
    topk = min(len(comb), topk_for_batch)
    comb = list(permutations(comb, topk))
    
    perm_pred = torch.stack([compute_prob(y_pred, c) for c in comb])
    perm_true = torch.stack([compute_prob(y_true, c) for c in comb])
    return F.kl_div(torch.log(perm_pred + eps), perm_true, reduction='none').sum()
    

def listNet_top_k(y_pred_, y_true_, topk_for_batch=DEFAULT_TOPK, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE):
    ### test
    # y_pred_ = torch.Tensor([[6, 2.9, 1, 0],[1, 2, 0, 2]]).requires_grad_(True)
    # y_true_ = torch.Tensor([[ 4.,  3.,  1., -1.], [ 2.,  2.,  1.,  0.]])
    ### test
    y_pred = y_pred_.clone()
    y_true = y_true_.clone()

    mask = y_true == padded_value_indicator
    y_pred[mask] = float('-inf')
    y_true[mask] = float('-inf')
    
    
    # comb = [i for i, m in enumerate(mask[0]) if not m]
    # topk = min(len(comb), topk_for_batch)
    # comb = list(permutations(comb, topk))
    
    # perm_pred = torch.stack([compute_prob(y_pred[0], c) for c in comb])
    # perm_true = torch.stack([compute_prob(y_true[0], c) for c in comb])
    
    loss = torch.stack([calculate_pair_kl_div(y_pred[j], y_true[j], mask[j], topk_for_batch, eps)
                              for j in range(len(y_pred))])
    return loss.mean()
    #print(F.kl_div(torch.log(perms_pred + eps), perms_true, reduction='none').sum())

    # return F.kl_div(torch.log(perm_pred + eps), perm_true, reduction='none').sum()
    
def listMLE(y_pred, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE):
    """
     introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    # shuffle for randomised tie resolution
    random_indices = torch.randperm(y_pred.shape[1])
    y_pred_shuffled = y_pred[:, random_indices]
    y_true_shuffled = y_true[:, random_indices]

    y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=1)

    mask = y_true_sorted == padded_value_indicator

    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
    preds_sorted_by_true[mask] = float("-inf")

    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max

    observation_loss[mask] = 0.0

    return torch.mean(torch.sum(observation_loss, dim=1))

# def STListNet(y_pred, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE):
#     '''
#     author = {Bruch, Sebastian and Han, Shuguang and Bendersky, Michael and Najork, Marc},
#     title = {A Stochastic Treatment of Learning to Rank Scoring Functions},
#     year = {2020},
#     booktitle = {Proceedings of the 13th International Conference on Web Search and Data Mining},
#     pages = {61–69}

#     The Top-1 approximated ListNet loss, which reduces to a softmax and simple cross entropy.
#     :param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents within a ltr_adhoc
#     :param batch_stds: [batch, ranking_size] each row represents the standard relevance grades for documents within a ltr_adhoc
#     :return:
#     '''

#     unif = torch.rand(batch_preds.size())  # [num_samples_per_query, ranking_size]
#     if self.gpu: unif = unif.to(self.device)

#     gumbel = -torch.log(-torch.log(unif + EPS) + EPS)  # Sample from gumbel distribution

#     batch_preds = (batch_preds + gumbel) / self.temperature

#     # todo-as-note: log(softmax(x)), doing these two operations separately is slower, and numerically unstable.
#     # c.f. https://pytorch.org/docs/stable/_modules/torch/nn/functional.html
#     batch_loss = torch.sum(-torch.sum(F.softmax(batch_stds, dim=1) * F.log_softmax(batch_preds, dim=1), dim=1))

#     self.optimizer.zero_grad()
#     batch_loss.backward()
#     self.optimizer.step()

#     return batch_loss

if __name__ == '__main__':
    # for test loss function, use combination (A,B,C,D) = 4! = 24
    # batch_size = 2, slate_size = 4
    y_true = torch.Tensor([[4,3,1,0],[2,2,1,0]])
    y_pred = torch.Tensor([[3.1,2.9,5,2],[1,2,0,2]])
    

    