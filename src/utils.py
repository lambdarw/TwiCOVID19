import datetime
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


class EarlyStopping():
    def __init__(self, path, patience=7, delta=0):
        self.path = path  # 最优模型保存路径
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            torch.save({'model': deepcopy(model.state_dict())}, self.path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            torch.save({'model': deepcopy(model.state_dict())}, self.path)
            self.counter = 0


def make_dataset(neu, pos, neg, time, args):
    # 使用滑动窗口，创建训练数据集
    tweet_df = pd.read_csv('./dataset/covid19_tweet.csv', sep='\t', index_col=None, header=0,
                           dtype={"id": str, "day": str, "created_at": str, "polarity": np.float32,
                                  "retweet_count": np.int32, "favorite_count": np.int32, "reply_count": np.int32,
                                  "quote_count": np.int32, "user_id": str, "reply_to_user_id": str,
                                  "user_mentions": str, "hashtags": str, "full_text": str})
    user_df = pd.read_csv('./dataset/covid19_user.csv', sep='\t', index_col=None, header=0,
                          dtype={"user_id": str, "followers_count": np.int32, "friends_count": np.int32,
                                 "listed_count": np.int32, "favourites_count": np.int32,
                                 "statuses_count": np.int32, "media_count": np.int32, "created_at": str,
                                 "name": str, "screen_name": str, "location": str, "description": str})
    topic_embed = np.load('./train_test/covid19_topic_embed.npy', allow_pickle=True).item()

    print('Preparing data...')
    out = []
    for i in tqdm(range(len(neg) - args.seq_len)):
        # x: (batch_size, seq_len, input_size)
        # y: (batch_size, output_size)
        x_neu = neu[i:i + args.seq_len]
        x_pos = pos[i:i + args.seq_len]
        x_neg = neg[i:i + args.seq_len]
        y_neu = neu[i + args.seq_len:i + args.seq_len + 1]
        y_pos = pos[i + args.seq_len:i + args.seq_len + 1]
        y_neg = neg[i + args.seq_len:i + args.seq_len + 1]
        x_neu = torch.FloatTensor(x_neu).view(args.batch_size, args.seq_len, args.input_size)
        x_pos = torch.FloatTensor(x_pos).view(args.batch_size, args.seq_len, args.input_size)
        x_neg = torch.FloatTensor(x_neg).view(args.batch_size, args.seq_len, args.input_size)
        y_neu = torch.FloatTensor(y_neu).view(args.batch_size, args.output_size)
        y_pos = torch.FloatTensor(y_pos).view(args.batch_size, args.output_size)
        y_neg = torch.FloatTensor(y_neg).view(args.batch_size, args.output_size)

        time_x = time[i:i + args.seq_len]

        # topic: (batch_size, seq_len, bert_size)
        topic = []
        for t in time_x:
            tmp = torch.zeros((args.bert_size), dtype=torch.float)
            for e in topic_embed[t]:
                tmp += torch.FloatTensor(e)
            topic.append(tmp)
        topic = torch.stack(topic, dim=0)
        topic = topic.unsqueeze(dim=0)

        # user_hist: [nums_user * [hist_len]] 不同user的hist_len不同，[hist_len]可能为空
        user_hist = []
        user_id_list = user_df['user_id'].tolist()
        start_date = datetime.datetime.strptime(time_x[0], '%Y-%m-%d')
        end_date = datetime.datetime.strptime(time_x[-1], '%Y-%m-%d')
        for i in range(len(user_id_list)):
            user_tweets = []
            user_id = user_id_list[i]
            tweets = tweet_df[tweet_df['user_id'] == user_id]
            for _, row in tweets.iterrows():
                tweet_id = row['id']
                tweet_date = datetime.datetime.strptime(row['day'], '%Y-%m-%d')
                if tweet_date < start_date or tweet_date > end_date:
                    continue
                user_tweets.append(tweet_id)
            user_hist.append(user_tweets)

        out.append((x_neu, x_pos, x_neg, y_neu, y_pos, y_neg, topic, user_hist))
    return out


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)
    mape = sum(np.abs((y_true - y_pred) / y_true)) / n
    return mape


def accuracy(trend_truth, trend_pred):
    acc = []
    for i in range(len(trend_truth)):
        if trend_truth[i] == trend_pred[i]:
            acc.append(1)
        else:
            acc.append(0)
    return np.mean(acc)
