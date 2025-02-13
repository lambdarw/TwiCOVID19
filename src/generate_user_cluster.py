import collections

import networkx as nx
import numpy as np
import pandas as pd
import torch
# PyEcharts V1.9.0
from pyecharts import options as opts
from pyecharts.charts import Scatter, Page
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_user_vec(user_df, pr):
    raw_attr_list = []
    raw_pr_list = []
    for idx, row in user_df.iterrows():
        attr = [row['followers_count'], row['friends_count'], row['listed_count'], row['favourites_count'],
                row['statuses_count'], row['media_count']]
        id = row['user_id']
        if id in pr:
            raw_pr_list.append([pr[id]])
        else:
            raw_pr_list.append([0])
        raw_attr_list.append(attr)
    attr_list = np.log2(np.array(raw_attr_list) + 1)
    pr_list = np.log2(np.array(raw_pr_list) * 100000 + 1)
    user_vec = np.column_stack((attr_list, pr_list))
    raw_user_vec = np.column_stack((raw_attr_list, raw_pr_list))
    return user_vec, raw_user_vec


def cal_pr(tweet_df):
    edges = []
    tweet_df_reply = tweet_df.dropna(axis=0, subset=['reply_to_user_id'])
    for idx, row in tweet_df_reply.iterrows():
        edges.append([row['user_id'], row['reply_to_user_id']])
    tweet_df_mention = tweet_df.dropna(axis=0, subset=['user_mentions'])
    for idx, row in tweet_df_mention.iterrows():
        mentions = row['user_mentions'].split(',')
        for m in mentions:
            edges.append([row['user_id'], m])

    G = nx.Graph()
    for n1, n2 in edges:
        G.add_edge(n1, n2)

    # PageRank
    pr = nx.pagerank(G)  # 返回一个字典，键是顶点，值是顶点的pr值
    return pr


def get_cluster_scatter(user_vec, k, cluster_id, is_selected):
    scatter = Scatter(init_opts=opts.InitOpts(width="800px", height="800px"))
    scatter.set_global_opts(
        title_opts=opts.TitleOpts(title="聚类散点图"),
        xaxis_opts=opts.AxisOpts(type_="value", is_show=False),
        yaxis_opts=opts.AxisOpts(type_="value", is_show=False),
    )
    for c in range(k):
        idx = [cluster_id[i] == c and is_selected[i] for i in range(len(cluster_id))]
        role_user_vec = user_vec[idx]
        x = role_user_vec[:, 0].tolist()
        y = role_user_vec[:, 1].tolist()
        scatter.add_xaxis(xaxis_data=x)
        scatter.add_yaxis(series_name='簇' + str(c), y_axis=y, symbol_size=5, label_opts=opts.LabelOpts(is_show=False))
    # 也可以这么写：
    # for c in range(k):
    #     x = []
    #     y = []
    #     for i in range(len(cluster_id)):
    #         if cluster_id[i] == c and is_selected[i]:
    #             x.append(user_vec[i, 0])
    #             y.append(user_vec[i, 1])
    #     scatter.add_xaxis(xaxis_data=x)
    #     scatter.add_yaxis(series_name='簇' + str(c), y_axis=y, symbol_size=5, label_opts=opts.LabelOpts(is_show=False))
    return scatter


def get_feature_scatter(user_df, k, cluster_id):
    scatter_list = []
    # ['followers_count', 'friends_count', 'listed_count', 'statuses_count', 'media_count']
    feature_name = ['followers_count', 'friends_count', 'listed_count', 'statuses_count', 'media_count']
    # [50, 20, 2, 100, 10]
    feature_step = [50, 20, 2, 100, 10]
    for feature_idx in range(len(feature_name)):
        name = feature_name[feature_idx]
        step = feature_step[feature_idx]
        scatter = Scatter(init_opts=opts.InitOpts(width="800px", height="800px"))
        scatter.set_global_opts(
            title_opts=opts.TitleOpts(title=f"{name}特征分布图"),
            xaxis_opts=opts.AxisOpts(type_="value"),
            yaxis_opts=opts.AxisOpts(type_="value"),
        )
        for c in range(k):
            user_idx = [i == c for i in cluster_id]
            role_user_df = user_df.loc[user_idx, :]

            dict = {}
            max = np.max(role_user_df[name])
            key_list = range(0, max + 1, step)
            dict[0] = 0
            for key in key_list:
                dict[key + step] = 0
            for _, row in role_user_df.iterrows():
                dict[int(np.ceil(row[name] / step)) * step] += 1
            new_dict = {np.log2(k): np.log2(v) for k, v in dict.items() if v != 0}

            scatter.add_xaxis(xaxis_data=list(new_dict.keys()))
            scatter.add_yaxis(series_name='簇' + str(c), y_axis=list(new_dict.values()), symbol_size=5,
                              label_opts=opts.LabelOpts(is_show=False))
        scatter_list.append(scatter)
    return scatter_list


def cluster(k):
    setup_seed(42)
    tweet_df = pd.read_csv('./dataset/covid19_tweet.csv', sep='\t', index_col=None, header=0,
                           dtype={"id": str, "day": str, "created_at": str, "polarity": np.float32,
                                  "retweet_count": np.int32, "favorite_count": np.int32, "reply_count": np.int32,
                                  "quote_count": np.int32, "user_id": str, "reply_to_user_id": str,
                                  "user_mentions": str,
                                  "hashtags": str, "full_text": str})
    user_df = pd.read_csv('./dataset/covid19_user.csv', sep='\t', index_col=None, header=0,
                          dtype={"user_id": str, "followers_count": np.int32, "friends_count": np.int32,
                                 "listed_count": np.int32, "favourites_count": np.int32,
                                 "statuses_count": np.int32, "media_count": np.int32, "created_at": str,
                                 "name": str, "screen_name": str, "location": str, "description": str})
    pr = cal_pr(tweet_df)
    user_vec, raw_user_vec = get_user_vec(user_df, pr)

    # KMeans聚类
    model = KMeans(n_clusters=k)
    model.fit(user_vec)
    y_pred = model.predict(user_vec).tolist()

    # 随机选取一部分用户
    is_selected = []
    for i in range(len(y_pred)):
        if np.random.random_sample() < 0.1:
            is_selected.append(True)
        else:
            is_selected.append(False)

    cluster_result = {
        'user_vec': user_vec,
        'raw_user_vec': raw_user_vec,
        'cluster_center': model.cluster_centers_,
        'cluster_id': y_pred,
        'is_selected': is_selected,
    }
    np.save(f'./train_test/covid19_cluster_k_{k}.npy', cluster_result)


def visualize(k):
    user_df = pd.read_csv('./dataset/covid19_user.csv', sep='\t', index_col=None, header=0,
                          dtype={"user_id": str, "followers_count": np.int32, "friends_count": np.int32,
                                 "listed_count": np.int32, "favourites_count": np.int32,
                                 "statuses_count": np.int32, "media_count": np.int32, "created_at": str,
                                 "name": str, "screen_name": str, "location": str, "description": str})
    cluster_result = np.load(f'./train_test/covid19_cluster_k_{k}.npy', allow_pickle=True).item()
    user_vec = cluster_result['user_vec']
    raw_user_vec = cluster_result['raw_user_vec']
    cluster_center = cluster_result['cluster_center']
    cluster_id = cluster_result['cluster_id']
    is_selected = cluster_result['is_selected']
    page = Page()

    # # 角色特征分布图
    # feature_scatter_list = get_feature_scatter(user_df, k, cluster_id)
    # for feature_scatter in feature_scatter_list:
    #     page.add(feature_scatter)
    # 角色特征统计量
    for c in range(k):
        idx = [i == c for i in cluster_id]
        role_raw_user_vec = raw_user_vec[idx]
        describe = pd.DataFrame(role_raw_user_vec).describe()
        print(describe)

    # 聚类散点图
    dec_user_vec = PCA(n_components=2).fit_transform(user_vec)
    cluster_scatter = get_cluster_scatter(dec_user_vec, k, cluster_id, is_selected)
    page.add(cluster_scatter)

    page.render(path=f'./visualize/covid19_cluster_k_{k}.html')


def count(k):
    cluster_result = np.load(f'./train_test/covid19_cluster_k_{k}.npy', allow_pickle=True).item()
    cluster_id = cluster_result['cluster_id']
    is_selected = cluster_result['is_selected']
    c = collections.Counter(cluster_id)  # 统计各类中用户数量
    d = collections.defaultdict(int)  # 统计每一类中选了多少个用户
    for i in range(len(cluster_id)):
        if is_selected[i]:
            d[cluster_id[i]] += 1
    print(f'k={k}，各类中用户数量={c}')
    print(f'k={k}，各类选取用户数量={d}')


def copy_backup(k):
    backup_cluster_result = np.load(f'./backup/covid19_cluster_k_{k}.npy', allow_pickle=True).item()
    cluster_result = np.load(f'./train_test/covid19_cluster_k_{k}.npy', allow_pickle=True).item()
    cluster_result['user_vec'] = backup_cluster_result['user_vec']
    cluster_result['cluster_center'] = backup_cluster_result['cluster_center']
    cluster_result['cluster_id'] = backup_cluster_result['cluster_id']
    cluster_result['is_selected'] = backup_cluster_result['is_selected']
    np.save(f'./train_test/covid19_cluster_k_{k}.npy', cluster_result)


def check_equal(k):
    print(f'k={k}')
    backup_cluster_result = np.load(f'./backup/covid19_cluster_k_{k}.npy', allow_pickle=True).item()
    cluster_result = np.load(f'./train_test/covid19_cluster_k_{k}.npy', allow_pickle=True).item()
    print((backup_cluster_result['user_vec'] == cluster_result['user_vec']).all())
    print((backup_cluster_result['cluster_center'] == cluster_result['cluster_center']).all())
    print(backup_cluster_result['cluster_id'] == cluster_result['cluster_id'])
    print(backup_cluster_result['is_selected'] == cluster_result['is_selected'])


if __name__ == '__main__':
    # cluster(k=1)
    # cluster(k=3)
    # cluster(k=5)
    # cluster(k=7)
    # cluster(k=9)
    # cluster(k=11)
    # cluster(k=13)
    # cluster(k=15)
    # count(k=1)
    # count(k=3)
    # count(k=5)
    # count(k=7)
    # count(k=9)
    # count(k=11)
    # count(k=13)
    # count(k=15)
    visualize(k=5)
    # copy_backup(k=3)
    # copy_backup(k=5)
    # copy_backup(k=7)
    # copy_backup(k=9)
    # check_equal(k=3)
    # check_equal(k=5)
    # check_equal(k=7)
    # check_equal(k=9)
