import numpy as np
from scipy.special import logsumexp

START = '<s>'
STOP = '</s>'

class CRFTagger:

    def __init__(self, conf):
        self.tag_to_idx = conf.tag_to_idx
        self.tag_size = conf.tag_size
        self.l2_alpha = conf.l2_alpha
        self.average = conf.average

    def update(self, samples, lr):
        gradients = np.zeros((self.feature_dim, self.tag_size))

        for sample in samples:
            self.biscores, self.start_score = self.get_biscores()
            word_ids, tag_ids = sample.word_ids, sample.tag_ids
            n = len(word_ids)
            for i in range(n):
                pre_tag_id = tag_ids[i - 1] if i > 0 else -1
                cur_tag_id = tag_ids[i]
                features = self.unigram_feature(word_ids, i) + self.bigram_feature(pre_tag_id)
                feat_ids = [self.feat_to_idx[feat] for feat in features if feat in self.feat_to_idx]
                for fd in feat_ids:
                    gradients[fd, cur_tag_id] += 1

            alpha = self.forward(word_ids)
            beta = self.backward(word_ids)
            logZ = logsumexp(alpha[-1])

            features = self.unigram_feature(word_ids, 0) + self.bigram_feature(-1)
            feat_ids = [self.feat_to_idx[feat] for feat in features if feat in self.feat_to_idx]
            uniscores = self.get_uniscores(word_ids, 0)
            prob = np.exp(beta[0] + self.start_score + uniscores - logZ)
            for fd in feat_ids:
                gradients[fd] -= prob

            for i in range(1, n):
                uniscores = self.get_uniscores(word_ids, i)
                mus = alpha[i - 1].reshape(-1, 1) + beta[i] + self.biscores + uniscores - logZ
                probs = np.exp(mus)
                feature = self.unigram_feature(word_ids, i)
                feat_ids = sorted([self.feat_to_idx[feat] for feat in feature if feat in self.feat_to_idx])

                for j in range(self.tag_size):
                    prob = probs[j]
                    for fd in feat_ids:
                        gradients[fd] -= prob
                    gradients[self.BF_IDX + j] -= prob

        # gradients -= self.l2_alpha * self.weights
        self.weights += lr * gradients

    def forward(self, word_ids):
        n = len(word_ids)
        alpha = np.zeros((n, self.tag_size))
        uniscores = self.get_uniscores(word_ids, 0)
        alpha[0] = uniscores + self.start_score

        for i in range(1, n):
            uniscores = self.get_uniscores(word_ids, i)
            scores = self.biscores + alpha[i - 1].reshape(-1, 1) + uniscores
            alpha[i] = logsumexp(scores, axis=0)

        return alpha

    def backward(self, word_ids):
        n = len(word_ids)
        beta = np.zeros((n, self.tag_size))
        uniscores = self.get_uniscores(word_ids, n - 1)
        for i in reversed(range(0, n - 1)):
            uniscores = self.get_uniscores(word_ids, i + 1)
            scores = self.biscores + beta[i + 1] + uniscores
            beta[i] = logsumexp(scores, axis=1)

        return beta

    def predict(self, word_ids):
        # viterbi decode
        n = len(word_ids)
        dp = np.zeros((n, self.tag_size))
        bkpt = np.zeros((n, self.tag_size), dtype=int)
        uniscores = self.get_uniscores(word_ids, 0)
        dp[0] = uniscores + self.start_score
        for i in range(1, n):
            uniscores = self.get_uniscores(word_ids, i)
            scores = self.biscores + dp[i - 1].reshape(-1, 1) + uniscores
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
        # maybe problem ?
        feature = self.unigram_feature(word_ids, t)
        feat_ids = [self.feat_to_idx[feat] for feat in feature if feat in self.feat_to_idx]
        feat_ids = np.array(feat_ids, dtype=int)
        uniscores = np.sum(self.weights[feat_ids], axis=0)
        return uniscores

    def get_biscores(self):
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

