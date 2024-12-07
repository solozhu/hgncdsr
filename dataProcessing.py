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

        self.batch_size = batch_size
        self.opt = opt
        self.eval = evaluation
        self.domains = domains

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

        if evaluation < 0:  
            self.train_data, self.train_user = self.read_train_data(source_train_data)
            data = self.preprocess()

        elif evaluation == 2:  
            self.test_data, self.test_user = self.read_test_data(source_valid_data)
            data = self.preprocess_for_predict()

        else: 
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
        item_number = 0
        with codecs.open(fname, "r", encoding="utf-8") as fr:
            for line in fr:
                item_number += 1
        return item_number

    def read_user(self, fname):
        user_number = 0
        with codecs.open(fname, "r", encoding="utf-8") as fr:
            for line in fr:
                user_number += 1
        return user_number

    def read_train_data(self, train_file):
        def takeSecond(elem):
            return elem[1]

        with codecs.open(train_file, "r", encoding="utf-8") as infile:
            train_data = []
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
                for r in res:
                    res_2.append(r[0]) 
                train_data.append(res_2)

        return train_data, user

    def read_test_data(self, test_file):
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


        processed = []

        for d, user in zip(self.train_data, self.train_user):  # the pad is needed! but to be careful.

            g = d[-1] 
            
            d = d[:-1]  
            xd = []  
            yd = [] 

            position = [0] * max_len
            x_position = [0] * max_len
            y_position = [0] * max_len

            for w in d:
                if w < self.opt.source_item_num:    
                    xd.append(w)   
                    yd.append(self.opt.source_item_num + self.opt.target_item_num)  
                  

                else: 
                    xd.append(self.opt.source_item_num + self.opt.target_item_num)  # pad
                    yd.append(w)


            if g >= self.opt.source_item_num:

                
                x_flag = 0
                y_flag = 1 
                yg = g     

                for id in range(len(xd)):
                    id += 1

                    if xd[-id] != self.opt.source_item_num + self.opt.target_item_num:
                        xg = xd[-id]
                        xd[-id] = self.opt.source_item_num + self.opt.target_item_num
                        break
            else: 
              
                x_flag = 1
                y_flag = 0
                xg = g
                for id in range(len(yd)):
                    id += 1
                    if yd[-id] != self.opt.source_item_num + self.opt.target_item_num:
                        yg = yd[-id]
                        yd[-id] = self.opt.source_item_num + self.opt.target_item_num
                        break
            
            if len(d) < max_len:
                xd = [self.opt.source_item_num + self.opt.target_item_num] * (max_len - len(d)) + xd
                yd = [self.opt.source_item_num + self.opt.target_item_num] * (max_len - len(d)) + yd
                d = [self.opt.source_item_num + self.opt.target_item_num] * (max_len - len(d)) + d
            
            index_x = 0  
            index_y = 0
            x_index = 0
            y_index = 0

            for id in range(max_len):
                id += 1
                if d[-id] != self.opt.source_item_num + self.opt.target_item_num:
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
            x_negative_sample = []
            for i in range(self.opt.neg_samples):
                while True:
                    sample = random.randint(0, self.opt.source_item_num - 1)
                    if sample != xg:
                        x_negative_sample.append(sample)
                        break
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
        if self.eval != -1: 
            batch = list(zip(*batch))
            return (torch.LongTensor(batch[0]), torch.LongTensor(batch[1]), torch.LongTensor(batch[2]),
                    torch.LongTensor(batch[3]), torch.LongTensor(batch[4]), torch.LongTensor(batch[5]),
                    torch.LongTensor(batch[6]), torch.LongTensor(batch[7]), torch.LongTensor(batch[8]),
                    torch.LongTensor(batch[9]), torch.LongTensor(batch[10]))
        else:           
            batch = list(zip(*batch))
            return (torch.LongTensor(batch[0]), torch.LongTensor(batch[1]), torch.LongTensor(batch[2]),
                    torch.LongTensor(batch[3]), torch.LongTensor(batch[4]), torch.LongTensor(batch[5]),
                    torch.LongTensor(batch[6]), torch.LongTensor(batch[7]), torch.LongTensor(batch[8]),
                    torch.LongTensor(batch[9]), torch.LongTensor(batch[10]), torch.LongTensor(batch[11]),
                    torch.LongTensor(batch[12]), torch.LongTensor(batch[13]), torch.LongTensor(batch[14]))


    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
