"""
    Data loader 
"""

import codecs
import random

import torch


class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """

    def __init__(self, domains, batch_size, opt, evaluation, predict_domain='x'):

        # 初始化一系列参数
        self.batch_size = batch_size
        self.opt = opt
        self.eval = evaluation
        self.domains = domains

        # 在处理验证集/测试集时所用的参数，指的是最终要预测用户在那个领域的下一步交互
        self.predict_domain = predict_domain  


        # ************* item_id *****************
        opt.source_item_num = self.read_item("./dataset/" + domains + "/Alist.txt")
        opt.target_item_num = self.read_item("./dataset/" + domains + "/Blist.txt")

        # ************* user id *****************
        opt.user_num = self.read_user("./dataset/" + domains + "/userlist.txt")

        # ************* sequential data *****************
        source_train_data = "./dataset/" + domains + "/traindata_new.txt"
        source_valid_data = "./dataset/" + domains + "/validdata_new2.txt"
        source_test_data = "./dataset/" + domains + "/testdata_new2.txt"

        if evaluation < 0:  # 构造训练集
            self.train_data, self.train_user = self.read_train_data(source_train_data)
            data = self.preprocess()

        elif evaluation == 2:   # 构造验证集
            self.test_data, self.test_user = self.read_test_data(source_valid_data)
            data = self.preprocess_for_predict()

        else:   # 构造测试集
            self.test_data, self.test_user = self.read_test_data(source_test_data)
            data = self.preprocess_for_predict()
        
        # shuffle for training
        if evaluation == -1:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
            if batch_size > len(data):
                batch_size = len(data)
                self.batch_size = batch_size
            if len(data) % batch_size != 0:
                data += data[:batch_size]
            data = data[: (len(data) // batch_size) * batch_size]

        # chunk into batches
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        self.data = data



    def read_item(self, fname):
        """
            统计 item 个数
        """
        item_number = 0
        with codecs.open(fname, "r", encoding="utf-8") as fr:
            for line in fr:
                item_number += 1
        return item_number

    def read_user(self, fname):
        """
            统计 user 个数
        """
        user_number = 0
        with codecs.open(fname, "r", encoding="utf-8") as fr:
            for line in fr:
                user_number += 1
        return user_number

    def read_train_data(self, train_file):
        """
            读取训练集
        """
        def takeSecond(elem):
            return elem[1]

        with codecs.open(train_file, "r", encoding="utf-8") as infile:
            train_data = []
            user = []
            for id, line in enumerate(infile):
                res = []
                line = line.strip().split("\t")
                user.append(int(line[0]))

                line = line[2:] # 交互的一系列物品
                for w in line:
                    w = w.split("|")
                    res.append((int(w[0]), int(w[1])))
                res.sort(key=takeSecond) # 按照时间顺序排列

                res_2 = []
                for r in res:
                    res_2.append(r[0])  # 只保留 item id
                train_data.append(res_2)

        return train_data, user

    def read_test_data(self, test_file):
        """
            读取验证or测试集
        """
        def takeSecond(elem):
            return elem[1]

        with codecs.open(test_file, "r", encoding="utf-8") as infile:
            test_data = []
            user = []
            for id, line in enumerate(infile):
                res = []
                line = line.strip().split("\t")
                user.append(int(line[0]))
                line = line[2:]
                for w in line:
                    w = w.split("|")
                    res.append((int(w[0]), int(w[1])))
                res.sort(key=takeSecond)

                res_2 = []
                for r in res[:-1]:
                    res_2.append(r[0])

                # denoted the corresponding validation/test entry
                if res[-1][0] >= self.opt.source_item_num:  
                    test_data.append([res_2, 1, res[-1][0]])
                else:
                    test_data.append([res_2, 0, res[-1][0]])
        return test_data, user

    def preprocess_for_predict(self):

        """ 与 preprocess 类似"""

        if "Enter" in self.domains:
            max_len = 30
            self.opt.maxlen = 30
            self.opt.L = 30
        elif 'Movie' in self.domains or 'Food' in self.domains:
            max_len = 15
            self.opt.maxlen = 15
            self.opt.L = 15
        else:
            max_len = 15
            self.opt.maxlen = 15
            self.opt.L = 15

        processed = []  
        for d, user in zip(self.test_data, self.test_user):  # the pad is needed! but to be careful.
            xd = []
            yd = []

            position = [0] * max_len
            x_position = [0] * max_len
            y_position = [0] * max_len

            x_flag = 1 - d[1]
            y_flag = d[1]

            if x_flag and self.predict_domain == 'y':
                continue
            if y_flag and self.predict_domain == 'x':
                continue

            for w in d[0]:
                # d[0] is test data
                if w < self.opt.source_item_num:
                    xd.append(w)
                    yd.append(self.opt.source_item_num + self.opt.target_item_num)

                else:
                    xd.append(self.opt.source_item_num + self.opt.target_item_num)
                    yd.append(w)

            if d[2] >= self.opt.source_item_num:
                # d[2] is ground truth, belongs to y domain
                for id in range(len(xd)):
                    id += 1
                    if xd[-id] != self.opt.source_item_num + self.opt.target_item_num:
                        xd[-id] = self.opt.source_item_num + self.opt.target_item_num
                        break
            else:
                # belongs to x domain
                for id in range(len(yd)):
                    id += 1
                    if yd[-id] != self.opt.source_item_num + self.opt.target_item_num:
                        yd[-id] = self.opt.source_item_num + self.opt.target_item_num
                        break

            if len(d[0]) < max_len:
                xd = [self.opt.source_item_num + self.opt.target_item_num] * (max_len - len(d[0])) + xd
                yd = [self.opt.source_item_num + self.opt.target_item_num] * (max_len - len(d[0])) + yd
                seq = [self.opt.source_item_num + self.opt.target_item_num] * (max_len - len(d[0])) + d[0]

            index_x = 0
            index_y = 0
            x_index = 0
            y_index = 0
            for id in range(max_len):
                id += 1
                if seq[-id] != self.opt.source_item_num + self.opt.target_item_num:
                    if seq[-id] < self.opt.source_item_num:
                        if xd[-id] != self.opt.source_item_num + self.opt.target_item_num:
                            x_index += 1
                            x_position[-id] = x_index
                        index_x += 1
                        position[-id] = index_x
                    else:
                        if yd[-id] != self.opt.source_item_num + self.opt.target_item_num:
                            y_index += 1
                            y_position[-id] = y_index
                        index_y += 1
                        position[-id] = index_y

            # 验证集/测试集 随机抽取负样本999个
            negative_sample = []
            for i in range(999):
                while True:
                    if d[1]:  # in Y domain, the validation/test negative samples
                        sample = random.randint(self.opt.source_item_num, self.opt.source_item_num + self.opt.target_item_num - 1)
                        if sample != d[2]:
                            negative_sample.append(sample)
                            break
                    else:  # in X domain, the validation/test negative samples
                        sample = random.randint(0, self.opt.source_item_num - 1)
                        if sample != d[2]:
                            negative_sample.append(sample)
                            break

            # seq: test data cross 
            # xd: test data for x domain
            # yd: test data for y domain
            # [d[2]]: ground truth
            # user: the user corresponding to the test data
            # x_flag: ground truth is in x domain
            # y_flag: ground truth is in y domain
            # negative_sample: negative sample 

            processed.append([seq, xd, yd, [d[2]], position, x_position, y_position, user, x_flag, y_flag, negative_sample])

        return processed
    def preprocess(self):
        
        def myprint(a):
            for i in a:
                print("%6d" % i, end="")
            print("")

        """ 构造成训练需要的格式 """
        if "Enter" in self.domains:
            max_len = 30
            self.opt.maxlen = 30
            self.opt.L = 30
        elif 'Movie' in self.domains or 'Food' in self.domains:
            max_len = 15
            self.opt.maxlen = 15
            self.opt.L = 15
        else:
            max_len = 15
            self.opt.maxlen = 15
            self.opt.L = 15


        processed = []

        for d, user in zip(self.train_data, self.train_user):  # the pad is needed! but to be careful.

            g = d[-1]  # ground truth，最新的交互物品即为需要预测的物品
            
            d = d[:-1]  # delete the ground truth，去除 ground truth 后的物品序列（跨域序列），跨域 train data
            xd = []  # x domain train data
            yd = []  # y domain train data

            # 位置序列初始化
            position = [0] * max_len
            x_position = [0] * max_len
            y_position = [0] * max_len

            # 构造单域训练数据
            for w in d:
                if w < self.opt.source_item_num:    # 说明该 item 属于 x domain
                    xd.append(w)    #   放到 x domain train data 里
                    yd.append(self.opt.source_item_num + self.opt.target_item_num)  
                    # pad，此位置 y domain train data用固定值填充

                else:   # 该 item 属于 y domain
                    xd.append(self.opt.source_item_num + self.opt.target_item_num)  # pad
                    yd.append(w)


            if g >= self.opt.source_item_num:
                # ground truth belongs to y domain
                
                x_flag = 0
                y_flag = 1  # flag 标记
                yg = g      # 跨域序列的预测目标和 y domain 序列的预测目标均为 g

                # 将 x domain 序列最近的 item 作为预测目标 ground truth
                for id in range(len(xd)):
                    id += 1

                    if xd[-id] != self.opt.source_item_num + self.opt.target_item_num:
                        # 从右到左看，找到第一个不是 pad 值的 item id，作为 x domain 的 ground truth
                        xg = xd[-id]
                        # 原 ground truth 位置改为 pad 值
                        xd[-id] = self.opt.source_item_num + self.opt.target_item_num
                        break
            else: 
                # ground truth belongs to x domain，下面的处理同理

                x_flag = 1
                y_flag = 0
                xg = g
                for id in range(len(yd)):
                    id += 1
                    if yd[-id] != self.opt.source_item_num + self.opt.target_item_num:
                        yg = yd[-id]
                        yd[-id] = self.opt.source_item_num + self.opt.target_item_num
                        break
            
            # 序列长度不足 max_len 的，用固定值填充补全序列长度
            if len(d) < max_len:
                xd = [self.opt.source_item_num + self.opt.target_item_num] * (max_len - len(d)) + xd
                yd = [self.opt.source_item_num + self.opt.target_item_num] * (max_len - len(d)) + yd
                d = [self.opt.source_item_num + self.opt.target_item_num] * (max_len - len(d)) + d
            
            # 构造位置序列
            index_x = 0     # 记录跨域中属于 x domain 的index
            index_y = 0     # 记录跨域中属于 y domain 的index
            x_index = 0     # 记录 x domain 的index
            y_index = 0     # 记录 x domain 的index

            for id in range(max_len):
                id += 1
                if d[-id] != self.opt.source_item_num + self.opt.target_item_num:
                    # 非 pad 值
                    if d[-id] < self.opt.source_item_num:
                        # x domain item
                        if xd[-id] != self.opt.source_item_num + self.opt.target_item_num:
                            x_index += 1
                            x_position[-id] = x_index
                        index_x += 1
                        position[-id] = index_x
                    else:
                        # y domain item
                        if yd[-id] != self.opt.source_item_num + self.opt.target_item_num:
                            y_index += 1
                            y_position[-id] = y_index
                        index_y += 1
                        position[-id] = index_y

            # 随机抽取负样本，这些 negative sample 与 ground truth 一起，模型为这些 item 打分做 top k 推荐
            negative_sample = []
            for i in range(self.opt.neg_samples):
                while True:
                    if g >= self.opt.source_item_num:  # ground truth in Y domain, the validation/test negative samples
                        sample = random.randint(self.opt.source_item_num, self.opt.source_item_num + self.opt.target_item_num - 1)
                        if sample != g:
                            negative_sample.append(sample)
                            break
                    else:  # ground truth in X domain, the validation/test negative samples
                        sample = random.randint(0, self.opt.source_item_num - 1)
                        if sample != g:
                            negative_sample.append(sample)
                            break
            # 同理为 x domain 抽取负样本
            x_negative_sample = []
            for i in range(self.opt.neg_samples):
                while True:
                    sample = random.randint(0, self.opt.source_item_num - 1)
                    if sample != xg:
                        x_negative_sample.append(sample)
                        break
            # 同理为 y domain 抽取负样本
            y_negative_sample = []
            for i in range(self.opt.neg_samples):
                while True:
                    sample = random.randint(self.opt.source_item_num,
                                            self.opt.source_item_num + self.opt.target_item_num - 1)
                    if sample != yg:
                        y_negative_sample.append(sample)
                        break

            # d: train data cross domain
            # xd: train data in x domain
            # yd: train data in y domain
            # [g]: ground truth for cross-domain
            # [xg]: ground truth for x domain
            # [yg]: ground truth for y domain
            # user: the user corresponding to the train data
            # x_flag: ground truth is in x domain
            # y_flag: ground truth is in y domain
            # negative_sample: negative samples 
            # x_negative_sample: negative samples for x domain
            # y_negative_sample: negative samples for y domain

            processed.append([d, xd, yd, [g], [xg], [yg], position, x_position, y_position, user, x_flag, y_flag, negative_sample, x_negative_sample,
                              y_negative_sample])
            

        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0:
            raise IndexError
        batch = self.data[key]
        if self.eval != -1:  # 验证集 or 测试集
            batch = list(zip(*batch))
            return (torch.LongTensor(batch[0]), torch.LongTensor(batch[1]), torch.LongTensor(batch[2]),
                    torch.LongTensor(batch[3]), torch.LongTensor(batch[4]), torch.LongTensor(batch[5]),
                    torch.LongTensor(batch[6]), torch.LongTensor(batch[7]), torch.LongTensor(batch[8]),
                    torch.LongTensor(batch[9]), torch.LongTensor(batch[10]))
        else:               # 训练集
            batch = list(zip(*batch))
            return (torch.LongTensor(batch[0]), torch.LongTensor(batch[1]), torch.LongTensor(batch[2]),
                    torch.LongTensor(batch[3]), torch.LongTensor(batch[4]), torch.LongTensor(batch[5]),
                    torch.LongTensor(batch[6]), torch.LongTensor(batch[7]), torch.LongTensor(batch[8]),
                    torch.LongTensor(batch[9]), torch.LongTensor(batch[10]), torch.LongTensor(batch[11]),
                    torch.LongTensor(batch[12]), torch.LongTensor(batch[13]), torch.LongTensor(batch[14]))

    # def __getitem__(self, key):
    #     if not isinstance(key, int):
    #         raise TypeError("Key must be an integer.")
    #     if key < 0:
    #         raise IndexError("Key must be non-negative.")
    #
    #     batch = self.data[key]
    #     batch = list(zip(*batch))  # 解包数据，将字段拆分为多个列表
    #
    #     # 需要统一长度的序列索引，例如：索引 2 到 10 是序列
    #     sequence_indices = range(2, 11) if self.eval != -1 else range(2, 15)
    #
    #     # 获取当前批次中序列的最大长度（跳过非序列字段）
    #     def safe_len(seq):
    #         return len(seq) if isinstance(seq, (list, tuple)) else 0
    #
    #     max_len = max(
    #         max(safe_len(seq) for seq in batch[idx]) for idx in sequence_indices
    #     )
    #
    #     # 填充或截断序列字段
    #     def pad_or_truncate(sequence, max_len, pad_value=0):
    #         """对序列进行填充或截断"""
    #         if not isinstance(sequence, (list, tuple)):
    #             return [pad_value] * max_len
    #         return sequence[:max_len] + [pad_value] * (max_len - len(sequence))
    #
    #     # 对需要填充的字段进行操作
    #     for idx in sequence_indices:
    #         batch[idx] = [pad_or_truncate(seq, max_len) for seq in batch[idx]]
    #
    #     # 将数据转换为 PyTorch 的 LongTensor
    #     if self.eval != -1:  # 验证集或测试集
    #         return tuple(torch.LongTensor(batch[i]) for i in range(11))
    #     else:  # 训练集
    #         return tuple(torch.LongTensor(batch[i]) for i in range(15))

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
