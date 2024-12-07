import torch
import torch.nn as nn

from torch.autograd import Variable


class HGN_CDR(nn.Module):
    def __init__(self,
                 model_args,
                 device):
        super(HGN_CDR, self).__init__()

        self.args = model_args

        # init args
        L = self.args.L
        dims = self.args.d
        item_num = self.args.item_num
        user_num = self.args.user_num

        # user and item embeddings
        self.user_embeddings = nn.Embedding(user_num, dims).to(device)
        self.item_embeddings = nn.Embedding(item_num, dims, padding_idx=item_num - 1).to(device)
        self.position_embeddings = nn.Embedding(L+1, dims, padding_idx=0).to(device)

        # feature gate
        self.feature_gate_user = nn.Linear(dims, dims).to(device)       

        self.feature_gate_item_src = nn.Linear(dims, dims).to(device)   
        self.feature_gate_item_tgt = nn.Linear(dims, dims).to(device) 
        self.feature_gate_item = nn.Linear(dims, dims).to(device)     

        # instance gate

        self.instance_gate_user = Variable(torch.zeros(dims, L).type(torch.FloatTensor),requires_grad=True).to(device)      
        self.instance_gate_position = Variable(torch.zeros(dims, 1).type(torch.FloatTensor), requires_grad=True).to(device)
        self.instance_gate_item_src = Variable(torch.zeros(dims, 1).type(torch.FloatTensor),requires_grad=True).to(device)
        self.instance_gate_item_tgt = Variable(torch.zeros(dims, 1).type(torch.FloatTensor),requires_grad=True).to(device)
        self.instance_gate_item = Variable(torch.zeros(dims, 1).type(torch.FloatTensor),requires_grad=True).to(device)

        
        self.instance_gate_user = torch.nn.init.xavier_uniform_(self.instance_gate_user)
        self.instance_gate_item_src = torch.nn.init.xavier_uniform_(self.instance_gate_item_src)
        self.instance_gate_item_tgt = torch.nn.init.xavier_uniform_(self.instance_gate_item_tgt)
        self.instance_gate_item = torch.nn.init.xavier_uniform_(self.instance_gate_item)
        self.instance_gate_position = torch.nn.init.xavier_uniform_(self.instance_gate_position)

        # items_to_predict embeddings
        self.W2 = nn.Embedding(item_num, dims, padding_idx=item_num - 1).to(device)
        self.b2 = nn.Embedding(item_num, 1, padding_idx=item_num - 1).to(device)

        # weight initialization
        self.user_embeddings.weight.data.normal_(0, 1.0 / self.user_embeddings.embedding_dim)
        self.item_embeddings.weight.data.normal_(0, 1.0 / self.item_embeddings.embedding_dim)
        self.position_embeddings.weight.data.normal_(0, 1.0 / self.position_embeddings.embedding_dim)

        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()

    def forward(self, seq, x_seq, y_seq, position, x_position, y_position, user_ids, items_to_predict, x_items_to_predict=None, y_items_to_predict=None, for_pred=False, pred_domain='x'):
        user_embs = self.user_embeddings(user_ids)
        w2 = self.W2(items_to_predict)
        b2 = self.b2(items_to_predict)

        item_embs = self.item_embeddings(seq)
        item_embs_src = self.item_embeddings(x_seq)
        item_embs_tgt = self.item_embeddings(y_seq)

        position_embs = self.position_embeddings(position)
        position_embs_src = self.position_embeddings(x_position)
        position_embs_tgt = self.position_embeddings(y_position)

        # src feature gating
        gate_src = torch.sigmoid(self.feature_gate_item_src(item_embs_src)
                                 + self.feature_gate_user(user_embs).unsqueeze(1))
        gated_item_src = item_embs_src * gate_src

        # tgt feature gating
        gate_tgt = torch.sigmoid(self.feature_gate_item_tgt(item_embs_tgt)
                                 + self.feature_gate_user(user_embs).unsqueeze(1))
        gated_item_tgt = item_embs_tgt * gate_tgt

        # cdr feature gating
        gate = torch.sigmoid(self.feature_gate_item(item_embs)
                             + self.feature_gate_user(user_embs).unsqueeze(1))
        gated_item = item_embs * gate

        # src instance gating
        instance_score_src = torch.sigmoid(
            torch.matmul(gated_item_src, self.instance_gate_item_src.unsqueeze(0)).squeeze()
            + user_embs.mm(self.instance_gate_user) 
            + torch.matmul(position_embs_src, self.instance_gate_position.unsqueeze(0)).squeeze()
            )
        union_out_src = gated_item_src * instance_score_src.unsqueeze(2)

        union_out_src = torch.sum(union_out_src, dim=1)
        union_out_src = union_out_src / torch.sum(instance_score_src, dim=1).unsqueeze(1)

        # tgt instance gating
        instance_score_tgt = torch.sigmoid(
            torch.matmul(gated_item_tgt, self.instance_gate_item_tgt.unsqueeze(0)).squeeze()
            + user_embs.mm(self.instance_gate_user) 
            + torch.matmul(position_embs_tgt, self.instance_gate_position.unsqueeze(0)).squeeze()
            )
        union_out_tgt = gated_item_tgt * instance_score_tgt.unsqueeze(2)

        union_out_tgt = torch.sum(union_out_tgt, dim=1)
        union_out_tgt = union_out_tgt / torch.sum(instance_score_tgt, dim=1).unsqueeze(1)

        # cdr instance gating
        instance_score = torch.sigmoid(
            torch.matmul(gated_item, self.instance_gate_item.unsqueeze(0)).squeeze()
            + user_embs.mm(self.instance_gate_user) 
            + torch.matmul(position_embs, self.instance_gate_position.unsqueeze(0)).squeeze()
            )
        union_out = gated_item * instance_score.unsqueeze(2)

        union_out = torch.sum(union_out, dim=1)
        union_out = union_out / torch.sum(instance_score, dim=1).unsqueeze(1)


        if for_pred:

            # cdr
            
            # # union level
            res = torch.bmm(union_out.unsqueeze(1), w2.permute(0,2,1)).squeeze()

            # item-item product
            rel_score = item_embs.bmm(w2.permute(0,2,1))
            rel_score = torch.mean(rel_score, dim=1)
            res += rel_score

            if pred_domain == 'x':
            # source domain
                
                # union level
                res_src = torch.bmm(union_out_src.unsqueeze(1), w2.permute(0, 2, 1)).squeeze()

                # item-item product
                rel_score_src = item_embs_src.bmm(w2.permute(0, 2, 1))
                rel_score_src = torch.mean(rel_score_src, dim=1)
                res_src += rel_score_src

                res += res_src 

            if pred_domain == 'y':
            # target domain

                # union level
                res_tgt = torch.bmm(union_out_tgt.unsqueeze(1), w2.permute(0, 2, 1)).squeeze()

                # item-item product
                rel_score_tgt = item_embs_tgt.bmm(w2.permute(0, 2, 1))
                rel_score_tgt = torch.mean(rel_score_tgt, dim=1)
                res_tgt += rel_score_tgt

                res += res_tgt


            return res

        else:

            w2_src = self.W2(x_items_to_predict)
            w2_tgt = self.W2(y_items_to_predict)

            # source domain
           
            # union level
            res_src = torch.bmm(union_out_src.unsqueeze(1), w2_src.permute(0, 2, 1)).squeeze()

            # item-item product
            rel_score_src = item_embs_src.bmm(w2_src.permute(0, 2, 1))
            rel_score_src = torch.mean(rel_score_src, dim=1)
            res_src += rel_score_src



            # target domain
            
            # union level
            res_tgt = torch.bmm(union_out_tgt.unsqueeze(1), w2_tgt.permute(0, 2, 1)).squeeze()

            # item-item product
            rel_score_tgt = item_embs_tgt.bmm(w2_tgt.permute(0, 2, 1))
            rel_score_tgt = torch.mean(rel_score_tgt, dim=1)
            res_tgt += rel_score_tgt



            # cdr
            
            # union level
            res = torch.bmm(union_out.unsqueeze(1), w2.permute(0, 2, 1)).squeeze()

            # item-item product
            rel_score = item_embs.bmm(w2.permute(0, 2, 1))
            rel_score = torch.mean(rel_score, dim=1)
            res += rel_score

            return res, res_src, res_tgt
