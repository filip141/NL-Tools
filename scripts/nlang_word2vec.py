import os
import random
import codecs
import logging
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.manifold import TSNE
from nlang_preprocess import WordTokenizer
from nlang_testing import split_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Word2Vec(object):
    def __init__(self, tokens, vec_dim=1000, context=5, neg_num=10, alpha=0.025, epoch=5, test_set_percent=0.3):
        self.alpha = alpha
        self.epoch = epoch
        self.neg_num = neg_num
        self.context = context
        self.tokens = tokens
        self.vec_dim = vec_dim
        self.word_set = set(tokens)
        self.vocab_words = len(self.word_set)
        self.test_set_percent = test_set_percent
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

    def train(self, no_test=False):
        test_set = None
        logger.info("Training model. Vocabulary length: {}, Hidden Layer: {}, Window: {}"
                    .format(self.vocab_words, self.vec_dim, self.context))
        # If not_test option is used
        if not no_test:
            training_set, test_set = split_dataset(self.test_set_percent, self.tokens)
        else:
            training_set = self.tokens
        # Iterate every epoch
        for epch in xrange(0, self.epoch):
            logger.info("Epoch: {}".format(epch))
            self.train_epoch(training_set)
            # Verify training on test set
            if test_set:
                accuracy, recall, precision, fscore = self.validate_epoch(test_set)
                logger.info("Accuracy: {}, Precision: {}, Recall: {}, Fscore: {}"
                            .format(accuracy, precision, recall, fscore))
        self.save()

    def validate_epoch(self, test_set):
        con_mat = np.zeros((2, 2))
        last_wnd = len(test_set) - self.context - 1
        for word_idx in xrange(self.context, last_wnd):
            # Set windows length
            start_idx = word_idx - self.context
            end_idx = word_idx + self.context + 1
            # Construct windows
            word_window = [self.word2id[word] for word in test_set[start_idx: end_idx]]
            word_center = word_window.pop((end_idx - start_idx) / 2)
            labels = [1.0] * (end_idx - start_idx - 1)

            # Get negative samples
            neg_samples = self.sample_unigram_dist(number=self.neg_num)
            labels = labels + [0.0] * self.neg_num
            cntx_words = word_window + neg_samples

            # Center word window
            vc = self.word_vec[word_center]
            vw = self.context_vec[cntx_words]
            sigmoid = 1. / (1. + np.exp(-np.dot(vw, vc.T)))
            for ll, sg in zip(labels, np.round(sigmoid)):
                con_mat[int(ll), int(sg)] += 1
        accuracy = (con_mat[0, 0] + con_mat[1, 1]) / np.sum(con_mat)
        acc_p = np.sum(con_mat, axis=1)
        acc_p[acc_p == 0] = 1
        recall = ((con_mat[0, 0] / acc_p[0]) + (con_mat[1, 1] / acc_p[1])) / 2.0
        pred = np.sum(con_mat, axis=0)
        pred[pred == 0] = 1
        precision = ((con_mat[0, 0] / pred[0]) + (con_mat[1, 1] / pred[1])) / 2.0
        fscore = 2 * ((precision * recall) / (precision + recall))
        return accuracy, recall, precision, fscore

    def train_epoch(self, training_set):
        last_wnd = len(training_set) - self.context - 1
        for word_idx in xrange(self.context, last_wnd):
            if word_idx % (last_wnd / 10) == 0:
                logger.info("Train Epoch Progress: {}%".format((word_idx / (last_wnd / 10)) * 10))
            # Set windows length
            start_idx = word_idx - self.context
            end_idx = word_idx + self.context + 1
            # Construct windows
            word_window = [self.word2id[word] for word in training_set[start_idx: end_idx]]
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
            lab_dif = (np.array(labels) - sigmoid) * self.alpha
            self.context_vec[cntx_words] += np.repeat(vc[np.newaxis, :], 20, axis=0) * lab_dif[:, np.newaxis]
            self.word_vec[word_center] += np.sum(vw * lab_dif[:, np.newaxis], axis=0)

    def save(self, path=None):
        if path is None:
            if not os.path.isdir("../data/model/"):
                os.mkdir("../data/model/")
            path = "../data/model/word2vec.npz"
        np.savez(path, neg_num=self.neg_num, context=self.context, vec_dim=self.vec_dim,
                 word2id=self.word2id, id2word=self.id2word, context_vec=self.context_vec,
                 word_vec=self.word_vec)

    def load(self, path=None):
        if path is None:
            path = "../data/model/word2vec.npz"
        with np.load(path) as X:
            self.neg_num = X["neg_num"]
            self.context = X["context"]
            self.vec_dim = X["vec_dim"]
            self.word2id = X["word2id"]
            self.id2word = X["id2word"]
            self.context_vec = X["context_vec"]
            self.word_vec = X["word_vec"]

    def plot(self):
        tsne = TSNE(n_components=2, random_state=0)
        all_vector_matrix = self.word_vec + self.context_vec
        all_vector_matrix_2d = tsne.fit_transform(all_vector_matrix)
        word_list = list(self.word_set)
        for i in xrange(len(word_list)):
            plt.text(all_vector_matrix_2d[i, 0], all_vector_matrix_2d[i, 1], word_list[i])
        plt.show()


if __name__ == '__main__':
    # Initialize preprocessing object
    word_token = WordTokenizer()
    # Read text file
    with codecs.open('../data/corpus.txt', 'r', encoding='utf8') as text_file:
        corpus = text_file.read()
    # Text to tokens
    token_list = word_token.text2tokens(corpus)
    w2v = Word2Vec(token_list, epoch=50, alpha=0.05)
    # w2v.load()
    # w2v.plot()
    w2v.train()
