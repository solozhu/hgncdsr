# HGNCDSR: Hierarchical Gating Network for Cross-Domain Sequential Recommendation

This repository contains the official implementation of the paper **"Hierarchical Gating Network for Cross-Domain Sequential 
Recommendation"**. 

paper link: https://doi.org/10.1145/3715321

Contact: jiabao@bit.edu.cn

Bibtex

```
@article{10.1145/3715321,
author = {Wang, Shuliang and Zhu, Jiabao and Wang, Yi and Ma, Chen and Zhao, Xin and Zhang, Yansen and Yuan, Ziqiang and Ruan, Sijie},
title = {Hierarchical Gating Network for Cross-Domain Sequential Recommendation},
year = {2025},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
issn = {1046-8188},
url = {https://doi.org/10.1145/3715321},
doi = {10.1145/3715321},
journal = {ACM Trans. Inf. Syst.},
month = mar
}
```

## Environments
- python 3.8
- PyTorch
- numpy
- scipy
- sklearn

## Dataset

The XXX_tem_sequences.pkl file is a list of lists that stores the inner item id of each user in a chronological order, e.g., user_records[0]=[item_id0, item_id1, item_id2,...].

The XXX_user_mapping.pkl file is a list that maps the user inner id to its original id.

The XXX_item_mapping.pkl file is similar to XXX_user_mapping.pkl.

## Example to run the code
Train and evaluate the model (you are strongly recommended to run the program on a machine with GPU):

python run.py
