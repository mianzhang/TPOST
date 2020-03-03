from .sample import Sample

# constants
START = "<s>"
STOP = "</s>"
UNK = "</unk>"

class Config:

    def __init__(self, args, logger):

        # data parameters
        self.train_file = '/'.join(['data', args.dataset, 'train.txt'])
        self.dev_file = '/'.join(['data', args.dataset, 'dev.txt'])
        self.test_file = '/'.join(['data', args.dataset, 'test.txt'])
        self.char_to_idx = {}
        self.idx_to_char = []
        self.tag_to_idx = {}
        self.idx_to_tag = []

        # training parameters
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.l2_alpha = args.l2_alpha
        self.lr_decay_rate = 0.96

        # model parameters
        self.model = args.model
        self.model_file = 'ckpt/' + '-'.join([self.model, args.dataset, 'model'])
        self.average = args.average
        if args.dataset == 'small':
            self.alpha = 0.01
        elif args.dataset == 'large':
            self.alpha = 0.01

    def build_dicts(self, train_data):
        charset, tagset = [], []
        for words, tags in train_data:
            for word, tag in zip(words, tags):
                for c in word:
                    charset.append(c)
                tagset.append(tag)
        charset = sorted(set(charset))
        tagset = sorted(set(tagset))
        for c in charset:
            self.char_to_idx[c] = len(self.char_to_idx)
        for tag in tagset:
            self.tag_to_idx[tag] = len(self.tag_to_idx)

        self.char_to_idx[UNK] = len(self.char_to_idx)
        self.UNK_INX = self.char_to_idx[UNK]
        self.idx_to_char.append(UNK)

       # for words, tags in train_data:
       #     for word, tag in zip(words, tags):
       #         for c in word:
       #             self.char_to_idx[c] = len(self.char_to_idx)
       #             self.idx_to_char.append(c)
       #         if tag not in self.tag_to_idx:
       #             self.tag_to_idx[tag] = len(self.tag_to_idx)
       #             self.idx_to_tag.append(tag)

        self.vocab_size = len(self.char_to_idx)
        self.tag_size = len(self.tag_to_idx)
        print('vocab size : ', self.vocab_size)
        print('tag size: ', self.tag_size)

    def map_to_ids(self, data):
        ids = []
        for words, tags in data:
            word_ids = []
            for word in words:
                word_id = []
                for c in word:
                    idx = self.char_to_idx[c] if c in self.char_to_idx else self.UNK_IDX
                    word_id.append(idx)
                word_ids.append(tuple(word_id))
                word_id = []
            tag_ids = [self.tag_to_idx[tag] for tag in tags]
            ids.append(Sample(word_ids, tag_ids))
        return ids



