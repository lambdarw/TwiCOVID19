import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

import model
from main.args import Args
from main.utils import EarlyStopping, make_dataset, setup_seed


def train(model, train_data, val_data, model_save_path, args):
    # 训练
    print('Training...')
    mse_loss = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    earlystopping = EarlyStopping(path=model_save_path)
    for i in tqdm(range(args.epoch)):
        # 训练
        model.train()
        train_loss = []
        for x_neu, x_pos, x_neg, y_neu, y_pos, y_neg, topic, user_hist in train_data:
            x_neu = x_neu.cuda()  # x_neu: (batch_size, seq_len, input_size)
            x_pos = x_pos.cuda()  # x_pos: (batch_size, seq_len, input_size)
            x_neg = x_neg.cuda()  # x_neg: (batch_size, seq_len, input_size)
            y_neu = y_neu.cuda()  # y_neu: (batch_size, output_size)
            y_pos = y_pos.cuda()  # y_pos: (batch_size, output_size)
            y_neg = y_neg.cuda()  # y_neg: (batch_size, output_size)
            topic = topic.cuda()  # topic: (batch_size, seq_len, bert_size)
            # user_hist: [nums_user * [hist_len]]

            if args.target_series == 'neu':
                x = x_neu
                y = y_neu
            elif args.target_series == 'pos':
                x = x_pos
                y = y_pos
            elif args.target_series == 'neg':
                x = x_neg
                y = y_neg
            else:
                raise Exception('Target series not found!')

            if args.model_name == 'Ours_U':
                p = model(x, topic)  # p: (batch_size, output_size)
            elif args.model_name == 'Ours_T':
                p = model(x, user_hist)  # p: (batch_size, output_size)
            elif args.model_name == 'Ours_S':
                p = model(topic, user_hist)  # p: (batch_size, output_size)
            elif args.model_name == 'Ours':
                p = model(x, topic, user_hist)  # p: (batch_size, output_size)
            elif args.model_name == 'Ours_v2':
                p = model(x, topic, user_hist)  # p: (batch_size, output_size)
            else:
                raise Exception('Model not found!')

            l = mse_loss(p, y)
            train_loss.append(l.item())

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        # 验证
        model.eval()
        val_loss = []
        truth = []
        pred = []
        trend_truth = []
        trend_pred = []
        for x_neu, x_pos, x_neg, y_neu, y_pos, y_neg, topic, user_hist in val_data:
            x_neu = x_neu.cuda()  # x_neu: (batch_size, seq_len, input_size)
            x_pos = x_pos.cuda()  # x_pos: (batch_size, seq_len, input_size)
            x_neg = x_neg.cuda()  # x_neg: (batch_size, seq_len, input_size)
            y_neu = y_neu.cuda()  # y_neu: (batch_size, output_size)
            y_pos = y_pos.cuda()  # y_pos: (batch_size, output_size)
            y_neg = y_neg.cuda()  # y_neg: (batch_size, output_size)
            topic = topic.cuda()  # topic: (batch_size, seq_len, bert_size)
            # user_hist: [nums_user * [hist_len]]

            if args.target_series == 'neu':
                x = x_neu
                y = y_neu
            elif args.target_series == 'pos':
                x = x_pos
                y = y_pos
            elif args.target_series == 'neg':
                x = x_neg
                y = y_neg
            else:
                raise Exception('Target series not found!')

            if args.model_name == 'Ours_U':
                p = model(x, topic)  # p: (batch_size, output_size)
            elif args.model_name == 'Ours_T':
                p = model(x, user_hist)  # p: (batch_size, output_size)
            elif args.model_name == 'Ours_S':
                p = model(topic, user_hist)  # p: (batch_size, output_size)
            elif args.model_name == 'Ours':
                p = model(x, topic, user_hist)  # p: (batch_size, output_size)
            elif args.model_name == 'Ours_v2':
                p = model(x, topic, user_hist)  # p: (batch_size, output_size)
            else:
                raise Exception('Model not found!')

            l = mse_loss(p, y)
            val_loss.append(l.item())
            truth.append(y.item())
            pred.append(p.item())
            trend_t = 1 if y.item() - x[:, -1, :].item() > 0 else 0
            trend_p = 1 if p.item() - x[:, -1, :].item() > 0 else 0
            trend_truth.append(trend_t)
            trend_pred.append(trend_p)
        print(f"Epoch {i}, train_loss = {np.mean(train_loss)}, val_loss = {np.mean(val_loss)}")
        # 早停
        # earlystopping(-accuracy(trend_truth, trend_pred), model)
        # earlystopping(mean_absolute_percentage_error(truth, pred), model)
        earlystopping(np.mean(val_loss), model)
        if earlystopping.early_stop:
            print("Early Stop.")
            break


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    setup_seed(42)

    # df = pd.read_csv('./dataset/AirPassengers.csv', sep=',', index_col=None, header=0)
    # time = df['date'].tolist()
    # series = df['value'].tolist()
    df = pd.read_csv('./train_test/covid19_series.csv', sep='\t', index_col=None, header=0)
    time = df['time'].tolist()
    neu = df['neu'].tolist()
    pos = df['pos'].tolist()
    neg = df['neg'].tolist()

    # 分割序列
    train_neu, test_neu = neu[:-(31 + args.seq_len)], neu[-(31 + args.seq_len):]
    train_pos, test_pos = pos[:-(31 + args.seq_len)], pos[-(31 + args.seq_len):]
    train_neg, test_neg = neg[:-(31 + args.seq_len)], neg[-(31 + args.seq_len):]
    train_time, test_time = time[:-(31 + args.seq_len)], time[-(31 + args.seq_len):]

    # 生成训练集、测试集
    train_data_path = f'./train_test/train_seq_len_{args.seq_len}.npy'
    test_data_path = f'./train_test/test_seq_len_{args.seq_len}.npy'
    if not os.path.exists(train_data_path):
        train_data = make_dataset(train_neu, train_pos, train_neg, train_time, args)
        np.save(train_data_path, {'train_data': train_data})
    if not os.path.exists(test_data_path):
        test_data = make_dataset(test_neu, test_pos, test_neg, test_time, args)
        np.save(test_data_path, {'test_data': test_data})
    train_data = np.load(train_data_path, allow_pickle=True).item()['train_data']
    test_data = np.load(test_data_path, allow_pickle=True).item()['test_data']

    model_save_path = f'./checkpoint/{args.model_name}_{args.target_series}_{args.seq_len}_{args.k}_{args.lr}.pth'
    print(f'Model save path: {model_save_path}')
    # 训练
    train_model = getattr(model, args.model_name, None)
    train_model = train_model(args).cuda()
    train(train_model, train_data, test_data, model_save_path, args)


if __name__ == '__main__':
    args = Args()
    main(args)
