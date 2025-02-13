class Args():
    batch_size = 1
    epoch = 100
    lr = 0.001  # todo

    input_size = 1
    output_size = 1
    bert_size = 768  # topic_embed的长度
    skep_size = 1024  # text_embed的长度

    model_name = 'Ours'  # todo
    target_series = 'neg'  # todo: ['neu', 'pos', 'neg']
    seq_len = 50  # 时间窗长度  # todo: [10, 30, 50]
    k = 15  # 聚类簇个数  # todo: [3, 5, 7, 9]
    gpu = '0'  # todo
