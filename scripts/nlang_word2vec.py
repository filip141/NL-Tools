import random
import codecs
import logging
import numpy as np
from collections import Counter
from nlang_preprocess import WordTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Word2Vec(object):
    def __init__(self, tokens, vec_dim=1000, context=5, neg_num=10, alpha=0.025, epoch=5):
        self.alpha = alpha
        self.epoch = epoch
        self.neg_num = neg_num
        self.context = context
        self.tokens = tokens
        self.vec_dim = vec_dim
        self.word_set = set(tokens)
        self.vocab_words = len(self.word_set)
        # Word 2 id conversion
        self.word2id = dict([(word, idx) for idx, word in enumerate(self.word_set)])
        self.id2word = dict([(idx, word) for idx, word in enumerate(self.word_set)])
        # Unigram Distribution
        self.counts, self.counts_dict = self.get_counts(tokens)
        self.unigram_cumsum = self.build_unigram_dist()
        # Initialize vectors
        self.word_vec = np.random.rand(self.vocab_words, vec_dim)
        self.context_vec = np.random.rand(self.vocab_words, vec_dim)
        self.word_vec /= (vec_dim * np.sum(self.word_vec, axis=1)[:, np.newaxis])
        self.context_vec /= (vec_dim * np.sum(self.context_vec, axis=1)[:, np.newaxis])

    def get_counts(self, tokens):
        count_occ = Counter(tokens)
        wset_list = [count_occ[self.id2word[idx]] for idx in xrange(self.vocab_words)]
        return wset_list, count_occ

    def build_unigram_dist(self):
        wset_cnts = np.array(self.counts)**0.75
        wset_arr = np.array(wset_cnts) / np.sum(wset_cnts)
        cumsum_wset = np.cumsum(wset_arr)
        return cumsum_wset

    def sample_unigram_dist(self, number=1):
        samples = []
        for _ in xrange(0, number):
            rnd = random.uniform(0, 1)
            # If small than first
            if rnd < self.unigram_cumsum[0]:
                samples.append(0)
                continue
            # for others
            counter = 0
            while rnd > self.unigram_cumsum[counter]:
                counter += 1
            samples.append(counter)
        return samples

    def train(self):
        logger.info("Training model. Vocabulary length: {}, Hidden Layer: {}, Window: {}"
                    .format(self.vocab_words, self.vec_dim, self.context))
        for epch in xrange(0, self.epoch):
            logger.info("Epoch: {}".format(epch))
            self.train_epoch()

    def train_epoch(self):
        diff_sum = 0
        last_wnd = len(self.tokens) - self.context - 1
        for word_idx in xrange(self.context, last_wnd):
            if word_idx % (last_wnd / 10) == 0:
                logger.info("Epoch Progress: {}%".format((word_idx / (last_wnd / 10)) * 10))
                logger.info("Vectors sum difference: {}, Log diff: {}".format(diff_sum, np.log(-diff_sum)))
            # Set windows length
            start_idx = word_idx - self.context
            end_idx = word_idx + self.context + 1
            # Construct windows
            word_window = [self.word2id[word] for word in self.tokens[start_idx: end_idx]]
            word_center = word_window.pop((end_idx - start_idx) / 2)
            labels = [1.0]*(end_idx - start_idx - 1)

            # Get negative samples
            neg_samples = self.sample_unigram_dist(number=self.neg_num)
            labels = labels + [0.0] * self.neg_num
            cntx_words = word_window + neg_samples

            # Center word window
            vc = self.word_vec[word_center]
            vw = self.context_vec[cntx_words]
            sigmoid = 1. / (1. + np.exp(-np.dot(vw, vc.T)))
            lab_dif = (sigmoid - np.array(labels)) * self.alpha
            self.context_vec[cntx_words] -= np.repeat(vc[np.newaxis, :], 20, axis=0) * lab_dif[:, np.newaxis]
            self.word_vec[word_center] -= np.sum(vw * lab_dif[:, np.newaxis], axis=0)
            diff_sum = diff_sum - (np.sum(self.context_vec) + np.sum(self.word_vec))


if __name__ == '__main__':
    # Initialize preprocessing object
    word_token = WordTokenizer()
    # Read text file
    with codecs.open('../data/corpus.txt', 'r', encoding='utf8') as text_file:
        corpus = text_file.read()
    # Text to tokens
    token_list = word_token.text2tokens(corpus)
    w2v = Word2Vec(token_list, epoch=30, alpha=0.05)
    w2v.train()
