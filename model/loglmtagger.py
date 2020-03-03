import numpy as np

class LogLMTagger:

    def __init__(self, conf):
        self.tag_to_idx = conf.tag_to_idx
        self.tag_size = conf.tag_size
        self.l2_alpha = conf.l2_alpha
        self.average = conf.average

    def update(self, samples, lr):
        gradients = np.zeros((self.feature_dim, self.tag_size))
        for sample in samples:
            word_ids, tag_ids = sample.word_ids, sample.tag_ids
            for i in range(len(word_ids)):
                scores, feat_ids = self.get_scores(word_ids, i)
                gold_tag_id = tag_ids[i]
                probs = np.exp(scores) / np.sum(np.exp(scores))
                for fd in feat_ids:
                    gradients[fd, gold_tag_id] += 1
                    gradients[fd] -= probs

        gradients -= self.l2_alpha * self.weights
        self.weights += lr * gradients
        if self.average:
            self.average_weights += self.weights

    def predict(self, word_ids):
        pred_tag_ids = []
        for i in range(len(word_ids)):
            scores, _ = self.get_scores(word_ids, i)
            pred_tag_ids.append(np.argmax(scores))
        return pred_tag_ids

    def create_feature_space(self, samples):
        feature_set = []
        for sample in samples:
            word_ids = sample.word_ids
            for i in range(len(word_ids)):
                feature_set += self.get_feature(word_ids, i)
        feature_set = set(feature_set)
        self.feat_to_idx = {feat: i for i, feat in enumerate(feature_set)}
        self.feature_dim = len(feature_set)
        self.weights = np.zeros((self.feature_dim, self.tag_size))
        if self.average:
            self.average_weights = np.zeros((self.feature_dim, self.tag_size))

    def get_scores(self, word_ids, t):
        feature = self.get_feature(word_ids, t)
        feat_ids = [self.feat_to_idx[feat] for feat in feature if feat in self.feat_to_idx]
        scores = np.sum(self.weights[feat_ids], axis=0)
        return scores, feat_ids

    def get_feature(self, word_ids, t):
        n = len(word_ids)
        feature = []
        pre_word = word_ids[t - 1] if t > 0 else '^^'
        cur_word = word_ids[t]
        next_word = word_ids[t + 1] if t < n - 1 else '$$'

        m = len(cur_word)
        feature.append(('02', cur_word))
        feature.append(('03', pre_word))
        feature.append(('04', next_word))
        feature.append(('05', cur_word, pre_word[-1]))
        feature.append(('06', cur_word, next_word[0]))
        feature.append(('07', cur_word[0]))
        feature.append(('08', cur_word[-1]))
        for i in range(1, m - 1):
            feature.append(('09', cur_word[i]))
            feature.append(('10', cur_word[0], cur_word[i]))
            feature.append(('11', cur_word[-1], cur_word[i]))
        if m == 1:
            feature.append(('12', cur_word, pre_word[-1], next_word[0]))
        for i in range(m):
            if i < m - 1 and cur_word[i] == cur_word[i + 1]:
                feature.append(('13', cur_word[i], 'consecutive'))
            if i < 4:
                feature.append(('14', cur_word[0: i + 1]))
            if m - i <= 4:
                feature.append(('15', cur_word[i: m]))

        return feature

