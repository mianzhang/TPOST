import numpy as np

START = '<s>'

class GLMTagger:

    def __init__(self, conf):
        self.tag_size = conf.tag_size
        self.average = conf.average

    def update(self, samples):
        for sample in samples:
            word_ids, tag_ids = sample.word_ids, sample.tag_ids
            n = len(word_ids)
            pred_tag_ids = self.predict(word_ids)
            if pred_tag_ids != tag_ids:
                for i in range(n):
                    feature = self.unigram_feature(word_ids, i)
                    feat_ids = [self.feat_to_idx[feat] for feat in feature if feat in self.feat_to_idx]
                    pre_tag_id = pred_tag_ids[i - 1] if i > 0 else -1
                    tag_id = pred_tag_ids[i]
                    for fd in feat_ids:
                        self.weights[fd, tag_id] -= 1
                        self.weights[self.BF_IDX + pre_tag_id, tag_id] -= 1
                    pre_tag_id = tag_ids[i - 1] if i > 0 else -1
                    tag_id = tag_ids[i]
                    for fd in feat_ids:
                        self.weights[fd, tag_id] += 1
                        self.weights[self.BF_IDX + pre_tag_id, tag_id] += 1
                if self.average:
                    self.average_weights += self.weights

    def predict(self, word_ids):
        # viterbi decode
        n = len(word_ids)
        dp = np.zeros((n, self.tag_size))
        bkpt = np.zeros((n, self.tag_size), dtype=int)
        uniscores = self.get_uniscores(word_ids, 0)
        biscores, start_score = self.get_biscores()
        dp[0] = uniscores + start_score
        for i in range(1, n):
            uniscores = self.get_uniscores(word_ids, i)
            scores = biscores + dp[i - 1].reshape(-1, 1) + uniscores
            dp[i] = np.amax(scores, axis=0)
            bkpt[i] = np.argmax(scores, axis=0)

        best_path = [np.argmax(dp[-1])]
        for i in reversed(range(1, n)):
            best_path.append(bkpt[i, best_path[-1]])
        best_path.reverse()

        return best_path

    def create_feature_space(self, samples):
        feature_set = []
        for sample in samples:
            word_ids = sample.word_ids
            for i in range(len(word_ids)):
                feature_set += self.unigram_feature(word_ids, i)

        feature_set = list(set(feature_set))
        self.BF_IDX = len(feature_set)
        for i in range(self.tag_size):
            feature_set += self.bigram_feature(i)
        feature_set += self.bigram_feature(-1)

        self.feat_to_idx = {feat: i for i, feat in enumerate(feature_set)}
        self.feature_dim= len(feature_set)
        print('The size of feature space: ', self.feature_dim)
        self.weights = np.zeros((self.feature_dim, self.tag_size))
        if self.average:
            self.average_weights = np.zeros((self.feature_dim, self.tag_size))

    def get_uniscores(self, word_ids, t):
        feature = self.unigram_feature(word_ids, t)
        feat_ids = [self.feat_to_idx[feat] for feat in feature if feat in self.feat_to_idx]
        feat_ids = np.array(feat_ids, dtype=int)
        if self.average:
            uniscores = np.sum(self.average_weights[feat_ids], axis=0)
        else:
            uniscores = np.sum(self.weights[feat_ids], axis=0)
        return uniscores

    def get_biscores(self):
        if self.average:
            return self.average_weights[self.BF_IDX: -1], self.weights[-1]
        else:
            return self.weights[self.BF_IDX: -1], self.weights[-1]

    def bigram_feature(self, pre_tag_id):
        return [('01', pre_tag_id)]

    def unigram_feature(self, word_ids, t):
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
