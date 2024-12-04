"""
    主运行程序
"""
import argparse
import datetime
import logging
from time import time

import numpy as np
import scipy.sparse as sp
import torch
from torch.autograd import Variable

from cdr_model import HGN_CDR
from dataProcessing import DataLoader
from evalMetrics import *

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def to_csr(user_ids, item_ids, num_users, num_items):
    # transform to a scipy.sparse CSR matrix
    row = user_ids
    col = item_ids
    data = np.ones(len(user_ids))
    coo = sp.coo_matrix((data, (row, col)), shape=(num_users, num_items))
    return coo.tocsr()


def negsamp_vectorized_bsearch_preverif(pos_inds, n_items, n_samp=31):
    """
        Pre-verified with binary search
        `pos_inds` is assumed to be ordered
        reference: https://tech.hbc.com/2018-03-23-negative-sampling-in-numpy.html
    """
    raw_samp = np.random.randint(0, n_items - len(pos_inds), size=n_samp)
    pos_inds_adj = pos_inds - np.arange(len(pos_inds))
    neg_inds = raw_samp + np.searchsorted(pos_inds_adj, raw_samp, side='right')
    return neg_inds


def generate_negative_samples(train_matrix, num_neg=3, num_sets=10):
    neg_samples = []
    for user_id, row in enumerate(train_matrix):
        pos_ind = row.indices
        neg_sample = negsamp_vectorized_bsearch_preverif(pos_ind, train_matrix.shape[1], num_neg * num_sets)
        neg_samples.append(neg_sample)
    return np.asarray(neg_samples).reshape(num_sets, train_matrix.shape[0], num_neg)


def evaluation(model, test_data_x, test_data_y, topk=10):
    pred_list = None
    ground_truth = None
    first_batch = True

    for batch in test_data_x:  # ground truth belongs to x domain

        if torch.cuda.is_available():
            inputs = [Variable(b.cuda()) for b in batch]
        else:
            inputs = [Variable(b) for b in batch]

        seq = inputs[0]  # cross-domain 序列
        x_seq = inputs[1]  # x domain 序列
        y_seq = inputs[2]  # y domain 序列
        ground = inputs[3]  # ground truth
        position = inputs[4]  # cross-domain 位置序列
        x_position = inputs[5]  # x domain 位置序列
        y_position = inputs[6]  # y domain 位置序列
        user = inputs[7]  # user
        x_flag = inputs[8]  # ground truth 属于 x domain, 1: yes, 0: no
        y_flag = inputs[9]  # ground truth 属于 y domain, 1: yes, 0: no
        neg = inputs[10]  # negative samples

        items_to_predict = torch.cat((ground, neg), 1)  # candidate items

        # candidate items' scores
        prediction_score = model(seq, x_seq, y_seq, position, x_position, y_position, user, items_to_predict,
                                 for_pred=True, pred_domain='x')

        prediction_score = prediction_score.cpu().data.numpy().copy()

        # get indexes of top-k scores
        # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
        ind = np.argpartition(prediction_score, -topk)
        ind = ind[:, -topk:]
        arr_ind = prediction_score[np.arange(len(prediction_score))[:, None], ind]
        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(prediction_score)), ::-1]
        batch_pred_list = ind[np.arange(len(prediction_score))[:, None], arr_ind_argsort]
        """
            for example
            模型为5个候选物品（包括1个 ground truth 和4个 negative items）打分，
            分别为：[0.8, 0.2, 0.5, 0.9, 0.7]
            那么，经过上面一段代码之后，batch_pred_list 为：[3, 0, 4, 2, 1]
            即 get indexes of top-k scores
        """

        if first_batch:
            pred_list = batch_pred_list
            ground_truth = ground.cpu().data.numpy().copy()
            flag = x_flag.cpu().data.numpy().copy()
            first_batch = False
        else:
            pred_list = np.append(pred_list, batch_pred_list, axis=0)
            ground_truth = np.append(ground_truth, ground.cpu().data.numpy().copy(), axis=0)
            flag = np.append(flag, x_flag.cpu().data.numpy().copy(), axis=0)

    for batch in test_data_y:  # ground truth belongs to x domain, 同理

        if torch.cuda.is_available():
            inputs = [Variable(b.cuda()) for b in batch]
        else:
            inputs = [Variable(b) for b in batch]
        seq = inputs[0]
        x_seq = inputs[1]
        y_seq = inputs[2]
        ground = inputs[3]
        position = inputs[4]
        x_position = inputs[5]
        y_position = inputs[6]
        user = inputs[7]
        x_flag = inputs[8]
        y_flag = inputs[9]
        neg = inputs[10]

        items_to_predict = torch.cat((ground, neg), 1)

        prediction_score = model(seq, x_seq, y_seq, position, x_position, y_position, user, items_to_predict,
                                 for_pred=True, pred_domain='y')
        prediction_score = prediction_score.cpu().data.numpy().copy()

        # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
        ind = np.argpartition(prediction_score, -topk)
        ind = ind[:, -topk:]
        arr_ind = prediction_score[np.arange(len(prediction_score))[:, None], ind]
        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(prediction_score)), ::-1]
        batch_pred_list = ind[np.arange(len(prediction_score))[:, None], arr_ind_argsort]

        if first_batch:
            pred_list = batch_pred_list
            ground_truth = ground.cpu().data.numpy().copy()
            flag = x_flag.cpu().data.numpy().copy()
            first_batch = False
        else:
            pred_list = np.append(pred_list, batch_pred_list, axis=0)
            ground_truth = np.append(ground_truth, ground.cpu().data.numpy().copy(), axis=0)
            flag = np.append(flag, x_flag.cpu().data.numpy().copy(), axis=0)

    # 计算指标
    HR, MRR, NDCG = [], [], []
    for k in [1, int(topk / 2), int(topk)]:
        HR.append(HR_at_k(pred_list, flag, k))
        MRR.append(MRR_(pred_list, flag))
        NDCG.append(ndcg_at_k(pred_list, flag, k))
    return HR, MRR, NDCG


def train_model(model, optim, train_data, valid_data_x, valid_data_y, test_data_x, test_data_y, conf):
    max_MRR_val_X = 0.0  # max MRR for valid dataset in X domain
    max_MRR_val_Y = 0.0  # max MRR for valid dataset in Y domain

    for epoch_num in range(conf.n_iter):
        t1 = time()

        # set model to training mode
        model.train()

        epoch_loss = 0.0
        num_batches = 0
        for batch in train_data:
            num_batches += 1

            if torch.cuda.is_available():
                inputs = [Variable(b.cuda()) for b in batch]
            else:
                inputs = [Variable(b) for b in batch]
            seq = inputs[0]  # cross-domain 序列
            x_seq = inputs[1]  # x domain 序列
            y_seq = inputs[2]  # y domain 序列
            ground = inputs[3]  # cross-domain ground truth
            x_ground = inputs[4]  # x domain ground truth
            y_ground = inputs[5]  # y domain ground truth
            position = inputs[6]  # cross-domain 位置序列
            x_position = inputs[7]  # x domain 位置序列
            y_position = inputs[8]  # y domain 位置序列
            user = inputs[9]  # 用户
            x_flag = inputs[10]  # ground truth 属于 x domain, 1: yes, 0: no
            y_flag = inputs[11]  # ground truth 属于 y domain, 1: yes, 0: no
            neg = inputs[12]  # negative items for cross-domain
            x_neg = inputs[13]  # negative items for x domain
            y_neg = inputs[14]  # negative items for y domain

            items_to_predict = torch.cat((ground, neg), 1)  # cross-domain 场景下的候选推荐物品
            x_items_to_predict = torch.cat((x_ground, x_neg), 1)  # x domain 候选推荐物品
            y_items_to_predict = torch.cat((y_ground, y_neg), 1)  # y domain 候选推荐物品

            # 根据模型，得到所有候选物品（目标推荐物品 ground truth 以及 negative items）的得分
            prediction_score, x_prediction_score, y_prediction_score = model(seq, x_seq, y_seq, position, x_position,
                                                                             y_position, user, items_to_predict,
                                                                             x_items_to_predict, y_items_to_predict,
                                                                             False)

            (aims_prediction, negatives_prediction) = torch.split(prediction_score, [ground.size(1), neg.size(1)],
                                                                  dim=1)
            (x_aims_prediction, x_negatives_prediction) = torch.split(x_prediction_score,
                                                                      [x_ground.size(1), x_neg.size(1)], dim=1)
            (y_aims_prediction, y_negatives_prediction) = torch.split(y_prediction_score,
                                                                      [y_ground.size(1), y_neg.size(1)], dim=1)

            # compute the BPR loss
            # cross-domain loss
            loss = -torch.log(torch.sigmoid(aims_prediction - negatives_prediction) + 1e-8)
            loss = torch.mean(torch.sum(loss))

            # x domain loss
            x_loss = -torch.log(torch.sigmoid(x_aims_prediction - x_negatives_prediction) + 1e-8)
            x_loss = torch.mean(torch.sum(x_loss))

            # y domain loss
            y_loss = -torch.log(torch.sigmoid(y_aims_prediction - y_negatives_prediction) + 1e-8)
            y_loss = torch.mean(torch.sum(y_loss))

            # final loss ,lamb用于平衡跨域和单域损失
            loss_all = conf.lamb * loss + (x_loss + y_loss) * (1 - conf.lamb)
            epoch_loss += loss_all.item()

            optim.zero_grad()
            loss_all.backward()
            optim.step()

        epoch_loss /= num_batches

        t2 = time()

        output_str = "Epoch %d [%.1f s]  loss=%.4f" % (epoch_num + 1, t2 - t1, epoch_loss)
        print(output_str)

        # evaluation，每 10 次迭代评估一次
        if (epoch_num + 1) % 10 == 0:

            model.eval()  # set model to evaluation mode

            HR, MRR, NDCG = evaluation(model, valid_data_x, valid_data_y, topk=10)

            MRR_val_X = MRR[0][0]
            MRR_val_Y = MRR[0][1]

            print("valid data evaluation: ---- MRR, NDCG@5, NDCG@10, HR@1, HR@5, HR@10")
            print([MRR[0], NDCG[1], NDCG[2], HR[0], HR[1], HR[2]])

            if MRR_val_X >= max_MRR_val_X or MRR_val_Y >= max_MRR_val_Y:
                HR, MRR, NDCG = evaluation(model, test_data_x, test_data_y, topk=10)

            if MRR_val_X >= max_MRR_val_X:
                max_MRR_val_X = MRR_val_X
                print("X best! ---- MRR, NDCG@5, NDCG@10, HR@1, HR@5, HR@10")
                print([MRR[0][0], NDCG[1][0], NDCG[2][0], HR[0][0], HR[1][0], HR[2][0]])
            if MRR_val_Y >= max_MRR_val_Y:
                max_MRR_val_Y = MRR_val_Y
                print("Y best! ---- MRR, NDCG@5, NDCG@10, HR@1, HR@5, HR@10")
                print([MRR[0][1], NDCG[1][1], NDCG[2][1], HR[0][1], HR[1][1], HR[2][1]])

    print("\n")
    print("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--L', type=int, default=15)  # 序列长度
    parser.add_argument('--d', type=int, default=256)  # embedding dimension
    parser.add_argument('--maxlen', type=int, default=15)  # 序列最大长度

    parser.add_argument('--n_iter', type=int, default=100)  # 迭代次数
    parser.add_argument('--seed', type=int, default=2040)  # 随机种子
    parser.add_argument('--batch_size', type=int, default=256)  # batch size
    parser.add_argument('--learning_rate', type=float, default=0.0005)  # learning rate
    parser.add_argument('--l2', type=float, default=1e-3)  # L2正则
    parser.add_argument('--neg_samples', type=int, default=99)  # 负样本抽样个数
    parser.add_argument('--lamb', type=float, default=0.8)  # loss 加权参数

    parser.add_argument('--data_dir', type=str, default='Food-Kitchen-50',
                        help='Food-Kitchen, Movie-Book, Entertainment-Education')  # domain

    config = parser.parse_args()

    # 构造用于训练、验证、测试的数据格式
    train_data = DataLoader(config.data_dir, config.batch_size, config, evaluation=-1)
    valid_data_x = DataLoader(config.data_dir, config.batch_size, config, evaluation=2, predict_domain='x')
    valid_data_y = DataLoader(config.data_dir, config.batch_size, config, evaluation=2, predict_domain='y')
    test_data_x = DataLoader(config.data_dir, config.batch_size, config, evaluation=1, predict_domain='x')
    test_data_y = DataLoader(config.data_dir, config.batch_size, config, evaluation=1, predict_domain='y')

    print("Data loading done!")

    # item num 总数，需要一个固定值用于填充，所以加一
    config.item_num = config.source_item_num + config.target_item_num + 1

    # 模型初始化
    model = HGN_CDR(config, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.l2)

    print("recommendation: -------------------------------------------------------------------")
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(config)
    print(device)

    # 训练
    train_model(model, optimizer, train_data, valid_data_x, valid_data_y, test_data_x, test_data_y, config)
