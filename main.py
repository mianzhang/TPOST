import os
import pickle
import argparse
import random
import logging
from datetime import datetime
import time
import numpy as np
from utils.config import Config
from utils.functions import *
from model.hmmtagger import HMMTagger
from model.lmtagger import LMTagger
from model.loglmtagger import LogLMTagger
from model.glmtagger import GLMTagger
from model.crftagger import CRFTagger

parser = argparse.ArgumentParser(description='POS tagging demo using HMM.')
parser.add_argument('--dataset', type=str, default='small', choices=['small', 'large'])
parser.add_argument('--model', type=str, default='HMM', choices=['HMM', 'LM', 'LogLM', 'GLM', 'CRF'])
parser.add_argument('--average', action='store_true')
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--l2_alpha', type=float, default=0.01)

s = 7
random.seed(s)
print('Set random seed number: ', s)

def _logging():
    os.mkdir(logdir)
    logfile = os.path.join(logdir, 'log.log')
    global logger
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    _format = logging.Formatter("%(name)-4s: %(levelname)-4s: %(message)s")
    fh.setFormatter(_format)
    ch.setFormatter(_format)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return

def logging_config(args):
    logger.info('Config:')
    for k, v in vars(args).items():
        logger.info(k + ": " + str(v))
    logger.info("")

def main(conf):
    train_data = read_data(conf.train_file)
    dev_data = read_data(conf.dev_file)
    if args.dataset == 'large':
        test_data = read_data(conf.test_file)

    conf.build_dicts(train_data)

    train_samples = conf.map_to_ids(train_data)
    dev_samples = conf.map_to_ids(dev_data)
    if args.dataset == 'large':
        test_samples = conf.map_to_ids(test_data)

    logger.info('number of train samples: %d' % len(train_samples))

    logger.info('number of dev samples: %d' % len(dev_samples))
    if args.dataset == 'large':
        logger.info('number of test samples: %d' % len(test_samples))

    if conf.model == 'HMM':
        tagger = HMMTagger(conf)
        tagger.train(train_samples)
    elif conf.model in ['LM', 'LogLM', 'GLM', 'CRF']:
        if conf.model == 'LM':
            print('Using LM')
            tagger = LMTagger(conf)
        if conf.model == 'LogLM':
            print('Using LogLM')
            tagger = LogLMTagger(conf)
        if conf.model == 'GLM':
            print('Using GLM')
            tagger = GLMTagger(conf)
        if conf.model == 'CRF':
            print('Using CRF')
            tagger = CRFTagger(conf)
        tagger.create_feature_space(train_samples)

        max_accuracy = 0.0
        n = len(train_samples)
        print("Train set size: ", n)
        for epoch in range(1, conf.epochs + 1):
            lr = conf.lr * (conf.lr_decay_rate)**(epoch - 1)
            start = time.time()
            logger.info('Epoch %d' % epoch)
            # train
            random.shuffle(train_samples)
            batchs = batching(train_samples, conf.batch_size)

            for batch in batchs:
                if conf.model in ['LM', 'GLM']:
                    tagger.update(batch)
                if conf.model in ['LogLM', 'CRF']:
                    tagger.update(batch, lr)

            # evaluate
            accuracy, total, tp = evaluate(tagger, train_samples, conf)
            logger.info('Train accuracy: %d / %d = %f' % (tp, total, accuracy))
            accuracy, total, tp = evaluate(tagger, dev_samples, conf)
            logger.info('Dev accuracy: %d / %d = %f' % (tp, total, accuracy))
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                with open(conf.model_file, 'wb') as f:
                    pickle.dump(tagger, f)
                    logger.info('Save best model.')
            end = time.time()
            logger.info('Elapse: %f' % (end - start))
            logger.info('')

    logger.info('')
    logger.info('Best:')
    with open(conf.model_file, 'rb') as f:
        tagger = pickle.load(f)
    accuracy, total, tp = evaluate(tagger, dev_samples, conf)
    logger.info('Dev accuracy: %d / %d = %f' % (tp, total, accuracy))
    if args.dataset == 'large':
        accuracy, total, tp = evaluate(tagger, test_samples, conf)
        logger.info('Test accuracy: %d / %d = %f' % (tp, total, accuracy))

if __name__ == '__main__':
    args = parser.parse_args()
    global logdir
    logdir = '-'.join([
        'log/log',
        args.dataset,
        args.model,
        datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    ])
    _logging()
    logging_config(args)
    conf = Config(args, logger)
    main(conf)




