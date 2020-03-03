import numpy as np


def read_data(data_file):
    data = []
    with open(data_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        words, tags = [], []
        for line in lines:
            line = line.strip().split()
            if len(line) == 0:
                data.append((words, tags))
                words, tags = [], []
                continue
            word, tag = line[1], line[3]
            words.append(word)
            tags.append(tag)
    return data

def evaluate(tagger, samples, conf):
    tp = 0
    total = 0
    for sample in samples:
        pred_path = tagger.predict(sample.word_ids)
        gold_path = sample.tag_ids
        total += len(gold_path)
        tp = tp + np.sum(np.array(pred_path) == np.array(gold_path))
    accuracy = 1.0 * tp / total * 100.0
    return accuracy, total, tp

def batching(samples, batch_size):
    batchs = []
    n = len(samples)
    if n % batch_size == 0:
        batch_nums = n // batch_size
    else:
        batch_nums = n // batch_size + 1
    for i in range(batch_nums):
        batchs.append(samples[i * batch_size: (i + 1) * batch_size])
    return batchs

