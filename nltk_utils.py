import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.tokenize.stanford_segmenter import StanfordSegmenter 
# nltk.download('punkt') #english tokenization
from nltk.parse import CoreNLPParser

parser = CoreNLPParser(url='http://localhost:9000')

stemmer = PorterStemmer()


def tokenize(sentence):
    return list(parser.tokenize(sentence))
    # return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag
