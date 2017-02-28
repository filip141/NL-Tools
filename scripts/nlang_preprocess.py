#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
import ast
import platform

if platform.system() == 'Windows':
    import morfeusz
else:
    import morfeusz2

emoticons_str = r"""
    (?:
        [:=;]
        [oO\-]?
        [D\)\]\(\]/\\OpP]
    )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    # r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]


class WordTokenizer(object):
    '''
        WordTokenizer divides each string into isolated
        words, this operation is performed using python
        Regular Expression
    '''

    def __init__(self, punfile="../data/punctation.txt", stopfile="../data/stopwords.txt"):
        # Initialize Morfeusz
        if platform.system() == 'Windows':
            self.morf = morfeusz
        else:
            self.morf = morfeusz2.Morfeusz()
        # Initialize files
        self.__punfile = punfile
        self.__stopfile = stopfile
        self.tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')',
                                    re.UNICODE | re.VERBOSE | re.IGNORECASE)
        self.emoticon_re = re.compile(r'^' + emoticons_str + '$',
                                      re.UNICODE | re.VERBOSE | re.IGNORECASE)
        self.undef_re = re.compile(r'^' + regex_str[-1] + '$',
                                   re.UNICODE | re.VERBOSE | re.IGNORECASE)
        self.men_re = re.compile(r'^' + regex_str[2] + '$',
                                 re.UNICODE | re.VERBOSE | re.IGNORECASE)
        self.url_re = re.compile(r'(' + '|'.join([regex_str[1], regex_str[4]]) + ')',
                                 re.UNICODE | re.VERBOSE | re.IGNORECASE)
        # Load list of punctation characters from file
        with open(self.__punfile) as punf:
            self.punctation = ast.literal_eval(punf.read())
        # Load list of stop words characters from file
        with open(self.__stopfile) as stopf:
            self.stop = ast.literal_eval(stopf.read())

    def tokenize(self, word):
        return self.tokens_re.findall(word)

    def preprocess(self, s, lowercase=False, words_only=False):
        tokens = self.tokenize(s)
        if words_only:
            tokens = [token
                      for token in tokens
                      if not self.emoticon_re.search(token)
                      and not self.url_re.search(token)
                      and not self.undef_re.search(token)
                      and not self.men_re.search(token)
                      ]
        # Lowercase option for words, not emoticon
        if lowercase:
            tokens = [token if self.emoticon_re.search(token) else token.lower() for token in tokens]
        return tokens

    def text2tokens(self, text):
        stop = self.stop + self.punctation + ["rt"]
        str_words = self.preprocess(text, lowercase=False,
                                    words_only=True)
        str_words = [term for term in str_words if term not in stop]
        an_words = []
        for mword in str_words:
            an_words.append(self.get_polish_letters(mword))
        return an_words

    @staticmethod
    def jaccard_similarity(x, y):
        intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
        union_cardinality = len(set.union(*[set(x), set(y)]))
        return intersection_cardinality / float(union_cardinality)

    def get_polish_letters(self, word):
        if platform.system() == 'Windows':
            try:
                sword = self.morf.analyse(word)
                return sword[0][0][1].lower()
            except KeyError:
                return word
        else:
            sword = self.morf.analyse(word)
            return sword[0][2][1].split(":")[0].lower()
