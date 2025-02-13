import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# PyEcharts V1.9.0
from pyecharts import options as opts
from pyecharts.charts import Line, Page
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import model
from main.args import Args
from main.utils import make_dataset, setup_seed, mean_absolute_percentage_error, accuracy


def predict(model, test_data, args):
    # 测试
    print('Predicting...')
    mse_loss = nn.MSELoss().cuda()
    model.eval()
    test_loss = []
    truth = []
    pred = []
    trend_truth = []
    trend_pred = []
    for x_neu, x_pos, x_neg, y_neu, y_pos, y_neg, topic, user_hist in test_data:
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
        test_loss.append(l.item())
        truth.append(y.item())
        pred.append(p.item())
        trend_t = 1 if y.item() - x[:, -1, :].item() > 0 else 0
        trend_p = 1 if p.item() - x[:, -1, :].item() > 0 else 0
        trend_truth.append(trend_t)
        trend_pred.append(trend_p)
    print(f"Test loss = {np.mean(test_loss)}")
    return truth, pred, trend_truth, trend_pred


def get_visual_line(title, xaxis_name, yaxis_name, time, truth, pred):
    line = Line()
    line.set_global_opts(
        xaxis_opts=opts.AxisOpts(
            name=xaxis_name,
            name_location='center',
            name_gap=40,
        ),
        yaxis_opts=opts.AxisOpts(
            name=yaxis_name,
            name_location='center',
            name_gap=40,
        ),
        title_opts=opts.TitleOpts(title=title),
        # datazoom_opts=opts.DataZoomOpts(is_show=True),
    )
    tmp_color = line.colors[0]
    line.colors[0] = 'blue'
    line.colors[1] = 'green'
    line.colors[2] = tmp_color
    line.add_xaxis(xaxis_data=time)
    line.add_yaxis(
        series_name='pred',
        y_axis=pred,
        label_opts=opts.LabelOpts(is_show=False),
        linestyle_opts=opts.LineStyleOpts(type_='dashed'),
        symbol='triangle',
        symbol_size=6,
        is_smooth=False,
    )
    line.add_yaxis(
        series_name='truth',
        y_axis=truth,
        label_opts=opts.LabelOpts(is_show=False),
        linestyle_opts=opts.LineStyleOpts(type_='dashed'),
        symbol='circle',
        symbol_size=6,
        is_smooth=False,
    )
    return line


def save_truth_pred(time, truth, pred):
    # 保存真实值和预测值
    df = pd.DataFrame({'time': time, 'truth': truth, 'pred': pred})
    df.to_excel('./visualize/covid19_truth_pred.xlsx', index=False)


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
    visual_save_path = f'./checkpoint/{args.model_name}_{args.target_series}_{args.seq_len}_{args.k}_{args.lr}.html'
    print(f'Model save path: {model_save_path}')
    # 预测
    test_model = getattr(model, args.model_name, None)
    test_model = test_model(args).cuda()
    test_model.load_state_dict(torch.load(model_save_path)['model'])
    truth, pred, trend_truth, trend_pred = predict(test_model, test_data, args)

    # 评价指标
    mae = mean_absolute_error(truth, pred)
    mse = mean_squared_error(truth, pred)
    mape = mean_absolute_percentage_error(truth, pred)
    acc = accuracy(trend_truth, trend_pred)
    print(f'mae={round(mae, 4)}, mse={round(mse, 4)}, mape={round(mape, 4)}, acc={round(acc, 4)}')

    # 可视化
    test_time = test_time[args.seq_len:]
    page = Page()
    title = f'舆论情感变化曲线图_{args.model_name}_{args.target_series}_{args.seq_len}_{args.k}_{args.lr}'
    visual_line = get_visual_line(title, '时间', args.target_series, test_time, truth, pred)
    page.add(visual_line)
    page.render(visual_save_path)

    # 保存真实值和预测值
    # save_truth_pred(test_time, truth, pred)

    return [round(mae, 4), round(mse, 4), round(mape, 4), round(acc, 4), visual_line]


if __name__ == '__main__':
    args = Args()
    main(args)
