import math


def HR_at_k(predicted, flag, topk):
    # 注意这里 predicted 里面都是index，不是具体的item ID
    sum_HR_x = 0.0
    num_x = 0.0
    sum_HR_y = 0.0
    num_y = 0.0
    num = len(predicted)
    for i in range(num):
        # 由于我们在候选物品中把 ground truth 放到了第一个位置上，
        # 所以其 index 为 0，所以我们只需要看top-k预测结果中是否有 0 ，有则命中
        act_set = set([0])
        pred_set = set(predicted[i][:topk])
        if flag[i]:
            num_x += 1
            sum_HR_x += len(act_set & pred_set)
        else:
            num_y += 1
            sum_HR_y += len(act_set & pred_set)

    return [sum_HR_x / num_x, sum_HR_y / num_y]


def MRR_(predicted, flag):
    res_x = 0.0
    res_y = 0.0
    num_x = 0.0
    num_y = 0.0
    for i in range(len(predicted)):
        pred_list = predicted[i]
        val = 0.0
        for j in range(0, len(pred_list)):
            if pred_list[j] == 0:
                val = 1.0 / (j + 1)
                break
        if flag[i]:
            num_x += 1
            res_x += val
        else:
            num_y += 1
            res_y += val
    return [res_x / num_x, res_y / num_y]


def ndcg_at_k(predicted, flag, topk):
    res_x = 0.0
    res_y = 0.0
    num_x = 0.0
    num_y = 0.0
    for i in range(len(predicted)):
        act_set = set([0])
        dcg_k = sum([int(predicted[i][j] in act_set) / math.log(j + 2, 2) for j in range(topk)])

        if flag[i]:
            num_x += 1
            res_x += dcg_k
        else:
            num_y += 1
            res_y += dcg_k
    return [res_x / num_x, res_y / num_y]
