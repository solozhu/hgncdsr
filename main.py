"""
    main.py: 保证验证集/测试集中的 user 在训练集中出现过，即非冷启动用户

    本实验直接采用 C2DSR 论文实验使用的数据集，包括 Food-Kitchen、Movie-Book、Entertainment-Education 三个场景：
    原数据集：
        Alist.txt
        Blist.txt
        testdata_new.txt
        testdata.txt
        traindata_new.txt
        traindata.txt
        userlist.txt
        validdata_new.txt
        validdata.txt
    main.py目的是在原数据集的基础上筛选，保证验证集/测试集中的 user 在训练集中出现过，即非冷启动用户，得到：
        testdata_new2.txt
        validdata_new2.txt

"""

import codecs

if __name__ == '__main__':

    path1 = "./dataset/processed/Movie-Book-20/traindata_new.txt"
    path2 = "./dataset/processed/Movie-Book-20/validdata_new2.txt"
    path3 = "./dataset/processed/Movie-Book-20/validdata_new2.txt"

    # 训练集出现的所有用户
    users = set()
    with codecs.open(path1, "r", encoding="utf-8") as infile:
        for id, line in enumerate(infile):
            line = line.strip().split("\t")
            u = int(line[0])
            users.add(u)

    # 筛选验证集/测试集的用户数据
    valid_data_new = []
    with codecs.open(path2, "r", encoding="utf-8") as infile:
        for id, line in enumerate(infile):
            line_temp = line.strip().split("\t")
            u = int(line_temp[0])
            if u in users:
                valid_data_new.append(line)

    # 保存
    with codecs.open(path3, "a", encoding="utf-8") as infile:
        for line in valid_data_new:
            infile.write(line)