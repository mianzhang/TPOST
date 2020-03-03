import numpy as np

class HMMTagger:

    def __init__(self, conf):
        self.tag_size = conf.tag_size
        self.vocab_size = conf.vocab_size
        self.alpha = conf.alpha

        self.start_count = np.zeros(self.tag_size)
        self.start_prob = np.zeros(self.tag_size)

        self.trans_count = np.zeros((self.tag_size, self.tag_size))
        self.trans_prob = np.zeros((self.tag_size, self.tag_size))

        self.stop_count = np.zeros(self.tag_size)
        self.stop_prob = np.zeros(self.tag_size)

        self.emit_count = np.zeros((self.tag_size, self.vocab_size))
        self.emit_prob = np.zeros((self.tag_size, self.vocab_size))

    def train(self, samples):
        # count
        for sample in samples:
            n = len(sample.tag_ids)
            tag_ids, word_ids = sample.tag_ids, sample.word_ids
            self.start_count[tag_ids[0]] += 1
            self.emit_count[tag_ids[0], word_ids[0]] += 1
            for i in range(1, n):
                pre_tag, cur_tag = tag_ids[i - 1], tag_ids[i]
                self.trans_count[pre_tag, cur_tag] += 1
                cur_word = word_ids[i]
                self.emit_count[cur_tag, cur_word] += 1
            self.stop_count[tag_ids[-1]] += 1
        # normalize to prob
        self.start_prob = self.add_k_smooth(self.start_count, self.alpha)
        self.trans_prob = self.add_k_smooth(self.trans_count, self.alpha)
        self.stop_prob = self.add_k_smooth(self.stop_count, self.alpha)
        self.emit_prob = self.add_k_smooth(self.emit_count, self.alpha)

    def add_k_smooth(self, count, alpha):
        prob = (count + alpha) / (np.sum(count, axis=-1, keepdims=True) + alpha * self.tag_size)
        return prob

    def predict(self, word_ids):
        # viterbi decode
        n = len(word_ids)
        dp = np.zeros((n, self.tag_size))
        bkpt = np.zeros((n, self.tag_size), dtype=int)

        dp[0] = np.log(self.start_prob) + np.log(self.emit_prob[:, word_ids[0]])
        for i in range(1, n):
            scores = np.log(self.trans_prob) + dp[i - 1].reshape(-1, 1) + np.log(self.emit_prob[:, word_ids[i]])
            dp[i] = np.amax(scores, axis=0)
            bkpt[i] = np.argmax(scores, axis=0)
        final = dp[-1] + np.log(self.stop_prob)

        best_path = [np.argmax(final)]
        for i in reversed(range(1, n)):
            best_path.append(bkpt[i, best_path[-1]])
        best_path.reverse()

        return best_path






