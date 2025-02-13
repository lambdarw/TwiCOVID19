import collections

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class Dense(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0., hasbias=True, act='relu'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.hasbias = hasbias
        if act == 'linear':
            self.act = self.linear_act
        elif act == 'relu':
            self.act = self.relu_act
        self.weights = nn.Parameter(torch.randn(input_dim, output_dim))
        if self.hasbias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_param()

    def linear_act(self, x):
        return x

    def relu_act(self, x):
        return F.relu(x)

    def reset_param(self):
        nn.init.xavier_uniform_(self.weights.data)

    def forward(self, inputs):
        x = inputs
        x = F.dropout(x, self.dropout)
        output = torch.matmul(x, self.weights)
        if self.hasbias:
            output += self.bias
        return self.act(output)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, mask=None):
        # queries: (batch_size, nums_q, d)
        # keys: (batch_size, nums_kv, d)
        # values: (batch_size, nums_kv, d_v)
        # 注：queries和keys的维度要相同，keys和values的数量要相同、维度可以不同
        d = queries.shape[-1]
        attn = torch.matmul(queries, keys.transpose(1, 2)) / np.sqrt(d)
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, values)  # output: (batch_size, nums_q, d_v)
        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d, d_v, h, dropout=0.1):
        # d_model: Output dimensionality of the model
        # d: Dimensionality of queries and keys
        # d_v: Dimensionality of values
        # h: Number of heads
        super().__init__()
        self.fc_q = nn.Linear(d_model, h * d)
        self.fc_k = nn.Linear(d_model, h * d)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d = d
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        # queries: (batch_size, nums_q, d_model)
        # keys: (batch_size, nums_kv, d_model)
        # values: (batch_size, nums_kv, d_model)
        # attention_mask: Mask over attention values (batch_size, h, nums_q, nums_kv). True indicates masking.
        # attention_weights: Multiplicative weights for attention values (batch_size, h, nums_q, nums_kv).
        # 注：
        # 当queries、keys、values为同一个矩阵时称作self-attention
        # 当queries、(keys、values)为来自不同的向量的矩阵变换时叫soft-attention

        batch_size, nums_q = queries.shape[:2]
        nums_kv = keys.shape[1]

        q = self.fc_q(queries) \
            .view(batch_size, nums_q, self.h, self.d).permute(0, 2, 1, 3)  # (batch_size, h, nums_q, d)
        k = self.fc_k(keys) \
            .view(batch_size, nums_kv, self.h, self.d).permute(0, 2, 3, 1)  # (batch_size, h, d, nums_kv)
        v = self.fc_v(values) \
            .view(batch_size, nums_kv, self.h, self.d_v).permute(0, 2, 1, 3)  # (batch_size, h, nums_kv, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d)  # (batch_size, h, nums_q, nums_kv)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v) \
            .permute(0, 2, 1, 3).contiguous().view(batch_size, nums_q, self.h * self.d_v)  # (batch_size, nums_q, h*d_v)
        out = self.fc_o(out)  # out: (batch_size, nums_q, d_model)
        return out, att


class LSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        hidden_size = 60
        self.lstm1 = nn.LSTM(args.input_size, hidden_size, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.linear1 = Dense(hidden_size, hidden_size, dropout=0.2, act='relu')
        self.linear2 = Dense(hidden_size, args.output_size, dropout=0., act='linear')

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        out, (h_n, c_n) = self.lstm1(x)  # out: (batch_size, seq_len, hidden_size)
        out, (h_n, c_n) = self.lstm2(out)  # out: (batch_size, seq_len, hidden_size)
        out = self.linear1(out)  # out: (batch_size, seq_len, hidden_size)
        out = self.linear2(out)  # out: (batch_size, seq_len, output_size)
        out = out[:, -1, :]  # 取最后一个时间步，out: (batch_size, output_size)
        return out


class GRU(nn.Module):
    def __init__(self, args):
        super().__init__()
        hidden_size = 60
        self.gru1 = nn.GRU(args.input_size, hidden_size, num_layers=1, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.linear1 = Dense(hidden_size, hidden_size, dropout=0.2, act='relu')
        self.linear2 = Dense(hidden_size, args.output_size, dropout=0., act='linear')

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        out, h_n = self.gru1(x)  # out: (batch_size, seq_len, hidden_size)
        out, h_n = self.gru2(out)  # out: (batch_size, seq_len, hidden_size)
        out = self.linear1(out)  # out: (batch_size, seq_len, hidden_size)
        out = self.linear2(out)  # out: (batch_size, seq_len, output_size)
        out = out[:, -1, :]  # 取最后一个时间步，out: (batch_size, output_size)
        return out


class Ours_U(nn.Module):
    def __init__(self, args):
        super().__init__()
        series_emb_size = 128
        topic_emb_size = 512
        hidden_size = 60
        self.series_emb = Dense(args.input_size, series_emb_size, dropout=0., act='relu')
        self.topic_emb = Dense(args.bert_size, topic_emb_size, dropout=0., act='relu')
        self.lstm1 = nn.LSTM(series_emb_size + topic_emb_size, hidden_size, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.linear1 = Dense(hidden_size, hidden_size, dropout=0.2, act='relu')
        self.linear2 = Dense(hidden_size, args.output_size, dropout=0., act='linear')

    def forward(self, x, topic):
        # x: (batch_size, seq_len, input_size)
        # topic: (batch_size, seq_len, bert_size)
        s = self.series_emb(x)  # s: (batch_size, seq_len, series_emb_size)
        t = self.topic_emb(topic)  # t: (batch_size, seq_len, topic_emb_size)
        out = torch.cat((s, t), dim=-1)  # out: (batch_size, seq_len, series_emb_size+topic_emb_size)
        out, (h_n, c_n) = self.lstm1(out)  # out: (batch_size, seq_len, hidden_size)
        out, (h_n, c_n) = self.lstm2(out)  # out: (batch_size, seq_len, hidden_size)
        out = self.linear1(out)  # out: (batch_size, seq_len, hidden_size)
        out = self.linear2(out)  # out: (batch_size, seq_len, output_size)
        out = out[:, -1, :]  # 取最后一个时间步，out: (batch_size, output_size)
        return out


class USER(nn.Module):
    def __init__(self, senti_size, d, d_v, h, args):
        # center_size: 聚类中心向量长度
        # senti_size: 情感向量长度（用户层==角色层==舆论层）
        super().__init__()
        self.skep_size = args.skep_size
        self.text_embed = np.load('./train_test/covid19_text_embed.npy', allow_pickle=True).item()
        self.cluster_result = np.load(f'./train_test/covid19_cluster_k_{args.k}.npy', allow_pickle=True).item()
        self.user_lstm = nn.LSTM(args.skep_size, senti_size, num_layers=1, batch_first=True)
        self.user_attention = ScaledDotProductAttention()
        self.role_attention = MultiHeadAttention(d_model=senti_size, d=d, d_v=d_v, h=h)
        self.save_count = 0

    def forward(self, user_hist):
        # user_hist: [nums_user * [hist_len]] 不同user的hist_len不同，[hist_len]可能为空

        # user_vec: (nums_user, n_features)
        # cluster_center: (n_clusters, n_features)
        # cluster_id: [nums_user]
        # is_selected: [nums_user]
        user_vec = torch.FloatTensor(self.cluster_result['user_vec']).cuda()
        cluster_center = torch.FloatTensor(self.cluster_result['cluster_center']).cuda()
        cluster_id = self.cluster_result['cluster_id']
        is_selected = self.cluster_result['is_selected']

        # 按角色对用户分类，并获取每个用户历史推文的embed
        user_feature = collections.defaultdict(list)  # user_feature: {role: [nums_role_user * (n_features)]}
        user_level = collections.defaultdict(list)  # user_level: {role: [nums_role_user * (hist_len, skep_size)]}
        for i in range(len(user_hist)):
            role_id = cluster_id[i]
            # 只选取一部分用户
            if not is_selected[i]:
                continue
            # 生成user_feature
            user_feature[role_id].append(user_vec[i, :])
            # 生成user_level
            hist = user_hist[i]
            if len(hist) == 0:
                hist = [torch.zeros((self.skep_size), dtype=torch.float)]
            else:
                hist = [torch.FloatTensor(self.text_embed[p]) for p in hist]
            hist = torch.stack(hist, dim=0)  # hist: (hist_len, skep_size)
            hist = hist.cuda()
            user_level[role_id].append(hist)
        # 用户历史推文的embed输入lstm
        for role_id, user_list in user_level.items():
            pack = nn.utils.rnn.pack_sequence(user_list, enforce_sorted=False)
            out, (h_n, c_n) = self.user_lstm(pack)
            h_n = h_n.squeeze(dim=0)  # h_n: (nums_role_user, senti_size)
            user_level[role_id] = h_n
        # 用户-->角色
        role_level = []  # role_level: [nums_role * (batch_size, senti_size)]
        user_attn = collections.defaultdict(int)  # 保存注意力系数
        for role_id, user_senti in user_level.items():
            center = cluster_center[role_id, :]  # center: (n_features)
            center = center.unsqueeze(dim=0).unsqueeze(dim=0)  # center: (batch_size, 1, n_features)
            user_fea = torch.stack(user_feature[role_id], dim=0)  # user_fea: (nums_role_user, n_features)
            user_fea = user_fea.unsqueeze(dim=0)  # user_fea: (batch_size, nums_role_user, n_features)
            user_senti = user_senti.unsqueeze(dim=0)  # user_senti: (batch_size, nums_role_user, senti_size)
            role_senti, attn = self.user_attention(center,
                                                   user_fea,
                                                   user_senti)  # role_senti: (batch_size, 1, senti_size)
            role_senti = role_senti.squeeze(dim=1)  # role_senti: (batch_size, senti_size)
            role_level.append(role_senti)
            user_attn[role_id] = attn
        # 角色-->舆论
        role_senti = torch.stack(role_level, dim=1)  # role_senti: (batch_size, nums_role, senti_size)
        role_senti, role_attn = self.role_attention(role_senti,
                                                    role_senti,
                                                    role_senti)  # role_senti: (batch_size, nums_role, senti_size)
        public_senti = role_senti.mean(dim=1)  # public_senti: (batch_size, senti_size)
        # self.save_attention_weight(user_attn, role_attn)  # 保存注意力系数
        return public_senti

    def save_attention_weight(self, user_attn, role_attn):
        # 保存注意力系数
        # copy到cpu上
        for k, v in user_attn.items():
            user_attn[k] = v[0, 0, :].detach().cpu().numpy()
        role_attn = role_attn[0, 0, 0, :].detach().cpu().numpy()
        # 求user_attn的最大长度，并填充至等长
        max_len = 0
        for v in user_attn.values():
            max_len = max(max_len, v.shape[0])
        for k, v in user_attn.items():
            user_attn[k] = np.pad(v, (0, max_len - v.shape[0]), 'constant', constant_values=(0, 0))
        # 保存
        df_user = pd.DataFrame(user_attn)
        df_user.to_excel(f'./visualize/covid19_attention_user_weight_{self.save_count}.xlsx', index=False)
        df_role = pd.DataFrame({'role_id': list(user_attn.keys()), 'role_attn': role_attn})
        df_role.to_excel(f'./visualize/covid19_attention_role_weight_{self.save_count}.xlsx', index=False)
        self.save_count += 1


class Ours_T(nn.Module):
    def __init__(self, args):
        super().__init__()
        hidden_size = 60
        senti_size = 128
        self.lstm1 = nn.LSTM(args.input_size, hidden_size, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.user = USER(senti_size, 64, 64, 2, args=args)
        self.linear1 = Dense(hidden_size + senti_size, hidden_size, dropout=0.2, act='relu')
        self.linear2 = Dense(hidden_size, args.output_size, dropout=0., act='linear')

    def forward(self, x, user_hist):
        # x: (batch_size, seq_len, input_size)
        # user_hist: [nums_user * [hist_len]] 不同user的hist_len不同，[hist_len]可能为空
        s, (h_n, c_n) = self.lstm1(x)  # s: (batch_size, seq_len, hidden_size)
        s, (h_n, c_n) = self.lstm2(s)  # s: (batch_size, seq_len, hidden_size)
        s = s[:, -1, :]  # 取最后一个时间步，s: (batch_size, hidden_size)
        u = self.user(user_hist)  # u: (batch_size, senti_size)
        out = torch.cat([s, u], dim=-1)  # out: (batch_size, hidden_size + senti_size)
        out = self.linear1(out)  # out: (batch_size, hidden_size)
        out = self.linear2(out)  # out: (batch_size, output_size)
        return out


class Ours_S(nn.Module):
    def __init__(self, args):
        super().__init__()
        topic_emb_size = 512
        hidden_size = 60
        senti_size = 128
        self.topic_emb = Dense(args.bert_size, topic_emb_size, dropout=0., act='relu')
        self.lstm1 = nn.LSTM(topic_emb_size, hidden_size, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.user = USER(senti_size, 64, 64, 2, args=args)
        self.linear1 = Dense(hidden_size + senti_size, hidden_size, dropout=0.2, act='relu')
        self.linear2 = Dense(hidden_size, args.output_size, dropout=0., act='linear')

    def forward(self, topic, user_hist):
        # x: (batch_size, seq_len, input_size)
        # topic: (batch_size, seq_len, bert_size)
        # user_hist: [nums_user * [hist_len]] 不同user的hist_len不同，[hist_len]可能为空
        t = self.topic_emb(topic)  # t: (batch_size, seq_len, topic_emb_size)
        t, (h_n, c_n) = self.lstm1(t)  # t: (batch_size, seq_len, hidden_size)
        t, (h_n, c_n) = self.lstm2(t)  # t: (batch_size, seq_len, hidden_size)
        t = t[:, -1, :]  # 取最后一个时间步，t: (batch_size, hidden_size)
        u = self.user(user_hist)  # u: (batch_size, senti_size)
        out = torch.cat([t, u], dim=-1)  # out: (batch_size, hidden_size + senti_size)
        out = self.linear1(out)  # out: (batch_size, hidden_size)
        out = self.linear2(out)  # out: (batch_size, output_size)
        return out


class Ours(nn.Module):
    def __init__(self, args):
        super().__init__()
        series_emb_size = 128
        topic_emb_size = 512
        hidden_size = 60
        senti_size = 128
        self.series_emb = Dense(args.input_size, series_emb_size, dropout=0., act='relu')
        self.topic_emb = Dense(args.bert_size, topic_emb_size, dropout=0., act='relu')
        self.lstm1 = nn.LSTM(series_emb_size + topic_emb_size, hidden_size, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.user = USER(senti_size, 64, 64, 2, args=args)
        self.linear1 = Dense(hidden_size + senti_size, hidden_size, dropout=0.2, act='relu')
        self.linear2 = Dense(hidden_size, args.output_size, dropout=0., act='linear')

    def forward(self, x, topic, user_hist):
        # x: (batch_size, seq_len, input_size)
        # topic: (batch_size, seq_len, bert_size)
        # user_hist: [nums_user * [hist_len]] 不同user的hist_len不同，[hist_len]可能为空
        s = self.series_emb(x)  # s: (batch_size, seq_len, series_emb_size)
        t = self.topic_emb(topic)  # t: (batch_size, seq_len, topic_emb_size)
        s_t = torch.cat((s, t), dim=-1)  # s_t: (batch_size, seq_len, series_emb_size+topic_emb_size)
        s_t, (h_n, c_n) = self.lstm1(s_t)  # s_t: (batch_size, seq_len, hidden_size)
        s_t, (h_n, c_n) = self.lstm2(s_t)  # s_t: (batch_size, seq_len, hidden_size)
        s_t = s_t[:, -1, :]  # 取最后一个时间步，s_t: (batch_size, hidden_size)
        u = self.user(user_hist)  # u: (batch_size, senti_size)
        out = torch.cat([s_t, u], dim=-1)  # out: (batch_size, hidden_size + senti_size)
        out = self.linear1(out)  # out: (batch_size, hidden_size)
        out = self.linear2(out)  # out: (batch_size, output_size)
        return out


class Ours_v2(nn.Module):
    def __init__(self, args):
        super().__init__()
        series_emb_size = 128
        topic_emb_size = 512
        hidden_size = 60
        senti_size = 128
        self.series_emb = Dense(args.input_size, series_emb_size, dropout=0., act='relu')
        self.topic_emb = Dense(args.bert_size, topic_emb_size, dropout=0., act='relu')
        self.lstm1 = nn.LSTM(series_emb_size + topic_emb_size, hidden_size, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.lstm_attention = MultiHeadAttention(d_model=hidden_size, d=64, d_v=64, h=1)
        self.user = USER(senti_size, d=64, d_v=64, h=2, args=args)
        self.linear1 = Dense(hidden_size + senti_size, hidden_size, dropout=0.2, act='relu')
        self.linear2 = Dense(hidden_size, args.output_size, dropout=0., act='linear')

    def forward(self, x, topic, user_hist):
        # x: (batch_size, seq_len, input_size)
        # topic: (batch_size, seq_len, bert_size)
        # user_hist: [nums_user * [hist_len]] 不同user的hist_len不同，[hist_len]可能为空
        s = self.series_emb(x)  # s: (batch_size, seq_len, series_emb_size)
        t = self.topic_emb(topic)  # t: (batch_size, seq_len, topic_emb_size)
        s_t = torch.cat((s, t), dim=-1)  # s_t: (batch_size, seq_len, series_emb_size+topic_emb_size)
        s_t, (h_n, c_n) = self.lstm1(s_t)  # s_t: (batch_size, seq_len, hidden_size)
        s_t, (h_n, c_n) = self.lstm2(s_t)  # s_t: (batch_size, seq_len, hidden_size)
        s_t = self.lstm_attention(s_t, s_t, s_t)  # s_t: (batch_size, seq_len, hidden_size)
        s_t = s_t.mean(dim=1)  # 加权平均，s_t: (batch_size, hidden_size)
        u = self.user(user_hist)  # u: (batch_size, senti_size)
        out = torch.cat([s_t, u], dim=-1)  # out: (batch_size, hidden_size + senti_size)
        out = self.linear1(out)  # out: (batch_size, hidden_size)
        out = self.linear2(out)  # out: (batch_size, output_size)
        return out
