import sys
import codecs
import numpy as np
from nlang_preprocess import WordTokenizer


class Word2Vec(object):
    def __init__(self, tokens):
        word_set = set(tokens)
        word2id = dict([(word, idx) for idx, word in enumerate(word_set)])

if __name__ == '__main__':
    # Initialize preprocessing object
    word_token = WordTokenizer()
    # Read text file
    with codecs.open('corpus.txt', 'r', encoding='utf8') as text_file:
        corpus = text_file.read()
    # Text to tokens
    token_list = word_token.text2tokens(corpus)
    w2v = Word2Vec(token_list)