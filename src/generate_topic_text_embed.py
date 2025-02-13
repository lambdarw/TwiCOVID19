import collections
import datetime
import json

import nltk
import numpy as np
import pandas as pd
import torch
from nltk.corpus import stopwords
# PyEcharts V1.9.0
from pyecharts import options as opts
from pyecharts.charts import WordCloud, Page
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from transformers import BertTokenizer, BertModel


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def text_process(s):
    # 分词
    s = nltk.word_tokenize(s)
    # 去掉停用词
    s = [w for w in s if (w not in stopwords.words('english'))]
    # 去掉长度小于等于3的词
    s = [w for w in s if (len(w) > 3)]
    return ' '.join(s)


def topic_model(docs):
    n_features = 1000  # 提取1000个特征词语
    tf_vectorizer = CountVectorizer(strip_accents='unicode', max_features=n_features, stop_words='english',
                                    max_df=1.0, min_df=5)
    tf = tf_vectorizer.fit_transform(docs)

    def find_k():
        min_k = 1
        max_k = 10
        res_k = 0
        min_perplexity = 10086
        for k in range(min_k, max_k):
            lda = LatentDirichletAllocation(n_components=k, max_iter=50, learning_method='batch', learning_offset=50,
                                            doc_topic_prior=0.1, topic_word_prior=0.01, random_state=0)
            lda.fit(tf)
            perplexity = lda.perplexity(tf)
            if perplexity < min_perplexity:
                res_k = k
                min_perplexity = perplexity
        print(f'topic model select k={res_k}')
        return res_k

    k = 7
    lda = LatentDirichletAllocation(n_components=k, max_iter=50, learning_method='batch', learning_offset=50,
                                    doc_topic_prior=0.1, topic_word_prior=0.01, random_state=0)
    lda.fit(tf)
    tf_feature_names = tf_vectorizer.get_feature_names()
    top_n = 25  # 每个topic下取25个词作为该topic的表示
    words = []
    weights = []
    for _, unsorted_weights in enumerate(lda.components_):
        idx = unsorted_weights.argsort()[:-top_n - 1:-1]
        for i in idx:
            words.append(tf_feature_names[i])
            weights.append(unsorted_weights[i])
    # words去重
    filter = collections.defaultdict(int)
    for i in range(len(words)):
        filter[words[i]] += weights[i]
    words = list(filter.keys())
    weights = list(filter.values())
    return words, weights


def get_bert_model(bert_path):
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    tokenizer.padding_side = 'right'
    bert = BertModel.from_pretrained(bert_path).cuda()
    return tokenizer, bert


def bert_embedding(text, tokenizer, bert):
    tokens = tokenizer(text, max_length=100, padding='max_length', truncation=True, return_tensors='pt')
    tokens = {k: v.cuda() for k, v in tokens.items()}
    out = bert(**tokens).pooler_output.squeeze(dim=0)
    return out.cpu().detach().numpy()


def topic_extract():
    file_path = './dataset/covid19_tweet.csv'
    df = pd.read_csv(file_path, sep='\t', index_col=None, header=0,
                     dtype={"id": str, "day": str, "created_at": str, "polarity": np.float32,
                            "retweet_count": np.int32, "favorite_count": np.int32, "reply_count": np.int32,
                            "quote_count": np.int32, "user_id": str, "reply_to_user_id": str, "user_mentions": str,
                            "hashtags": str, "full_text": str})
    df['processed_text'] = df['full_text'].apply(text_process)
    # 提取主题词
    group = df.groupby('day')
    day_topic = {}  # day_topic: {day: ([word, ...], [weight, ...])}
    for key, value in group:
        topic = topic_model(value['processed_text'])
        day_topic[key] = topic
    # 统计出现频率较高的词
    all_words = []
    for day, (words, weights) in day_topic.items():
        all_words += words
    c = collections.Counter(all_words)
    common_words = {word for word, freq in c.most_common(5)}
    # 去掉出现频率较高的词
    for day, (words, weights) in day_topic.items():
        filter = collections.defaultdict(int)
        for k, v in zip(words, weights):
            filter[k] += v
        for common_word in common_words:
            if common_word in filter:
                filter.pop(common_word)
        day_topic[day] = (list(filter.keys()), list(filter.values()))
    # 保存
    with open('./train_test/covid19_topic.json', 'w') as f:
        json.dump(day_topic, f)
    return day_topic


def topic_embed(day_topic):
    # 生成主题词的词嵌入
    tokenizer, bert = get_bert_model(bert_path='../bert-base-uncased')
    topic_embed = {}
    for day, topic in day_topic.items():
        topic_embed[day] = [bert_embedding(word, tokenizer, bert) for word in topic[0]]
    np.save('./train_test/covid19_topic_embed.npy', topic_embed)


def text_embed():
    # 生成文本的词嵌入
    tokenizer, bert = get_bert_model(bert_path='../skep-ernie2-bert-large')
    file_path = './dataset/covid19_tweet.csv'
    df = pd.read_csv(file_path, sep='\t', index_col=None, header=0,
                     dtype={"id": str, "day": str, "created_at": str, "polarity": np.float32,
                            "retweet_count": np.int32, "favorite_count": np.int32, "reply_count": np.int32,
                            "quote_count": np.int32, "user_id": str, "reply_to_user_id": str, "user_mentions": str,
                            "hashtags": str, "full_text": str})
    text_embed = {}
    for _, row in df.iterrows():
        id = row['id']
        text = row['full_text']
        text_embed[id] = bert_embedding(text, tokenizer, bert)
    np.save('./train_test/covid19_text_embed.npy', text_embed)


def get_wordcloud(title, words, value):
    data = zip(words, value)
    wordcloud = WordCloud()
    WordCloud()
    wordcloud.add(series_name="词云", data_pair=data, word_size_range=[6, 66])
    wordcloud.set_global_opts(
        title_opts=opts.TitleOpts(title=title, title_textstyle_opts=opts.TextStyleOpts(font_size=23)),
        tooltip_opts=opts.TooltipOpts(is_show=True),
    )
    return wordcloud


def draw_topic_wordcloud(start_date, end_date):
    file_path = './dataset/covid19_tweet.csv'
    df = pd.read_csv(file_path, sep='\t', index_col=None, header=0,
                     dtype={"id": str, "day": str, "created_at": str, "polarity": np.float32,
                            "retweet_count": np.int32, "favorite_count": np.int32, "reply_count": np.int32,
                            "quote_count": np.int32, "user_id": str, "reply_to_user_id": str, "user_mentions": str,
                            "hashtags": str, "full_text": str})
    df['processed_text'] = df['full_text'].apply(text_process)
    # 按时间筛选df
    start_day = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_day = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    docs = []
    for _, row in df.iterrows():
        tweet_day = datetime.datetime.strptime(row['day'], '%Y-%m-%d')
        if tweet_day < start_day or tweet_day > end_day:
            continue
        docs.append(row['processed_text'])
    # 提取主题词
    words, weights = topic_model(docs)
    page = Page()
    # 绘制词云
    raw_wordcloud = get_wordcloud(title='原始词云', words=words, value=weights)
    page.add(raw_wordcloud)
    # 去掉权重top3的词
    idx = np.array(weights).argsort()[::-1]
    words_sorted = [words[i] for i in idx][3:]
    weights_sorted = [weights[i] for i in idx][3:]
    # 绘制词云
    filter_wordcloud = get_wordcloud(title='筛选后词云', words=words_sorted, value=weights_sorted)
    page.add(filter_wordcloud)
    page.render(f'./visualize/covid19_topic_wordcloud_{start_date}_{end_date}.html')


if __name__ == '__main__':
    setup_seed(42)
    # day_topic = topic_extract()
    # topic_embed(day_topic)
    # text_embed()
    draw_topic_wordcloud(start_date='2021-12-04', end_date='2021-12-05')
    draw_topic_wordcloud(start_date='2021-12-25', end_date='2021-12-27')
