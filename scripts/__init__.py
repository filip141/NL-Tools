import logging

try:
    from nlang_preprocess import WordTokenizer
except ImportError:
    logging.exception("Import error, verify your installation.")
