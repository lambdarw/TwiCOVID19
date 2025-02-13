class Args():
    batch_size = 1
    epoch = 100
    lr = 0.001  # todo

    input_size = 1
    output_size = 1
    bert_size = 768  # topic_embed
    skep_size = 1024  # text_embed

    model_name = 'Ours'
    target_series = 'neg'  # todo: ['neu', 'pos', 'neg']
    seq_len = 50  # time window length  # todo: [10, 30, 50]
    k = 3  # clusters number  # todo: [3, 5, 7, 9]
    gpu = '0'
