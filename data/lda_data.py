from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import random
import pickle
import os

class Hyperparams:
    def __init__(self):
        self.iterations = 100 
        self.topics = 20 
        self.alpha = 0.01 
        self.gamma = 0.01

        #self.doc_cnt = len(docs)
        #self.wrd_cnt = len(dictionary)

class DataLoader:
    def __init__(self):
        self.data = self.load_data()
        self.processed_docs = []

    def load_data(self):
        return fetch_20newsgroups(subset='train').data

    def preprocess(self):
        path = "data/processed.pickle"
        if os.path.exists(path):
            print("Loading existing preprocessed documents...")
            with open(path, "rb") as file:
                self.processed_docs = pickle.load(file)
        else:
            print("Existing preprocessed documents not found. Preprocessing now...")
            self.processed_docs = list(map(self.preprocess_doc, self.data))
            with open(path, "wb") as file:
                pickle.dump(self.processed_docs, file)
            print("Done preprocessing. Saving data to " + path)

    # Preprocess documents - lemmatization and stemming
    def lemmatize_stemming(self, text):
        stemmer = SnowballStemmer("english")
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

    def preprocess_doc(self, text):
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                result.append(self.lemmatize_stemming(token))
        return result

    def build_dictionary(self):
        # Construct dictionary
        dictionary = gensim.corpora.Dictionary(self.processed_docs)
        dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
        return dictionary

    def filter_docs(self, dictionary):
        # Filter words in documents
        docs = list()
        maxdoclen = 0 
        for doc in self.processed_docs:
            docs.append(list(filter(lambda x: x != -1, dictionary.doc2idx(doc))))
            maxdoclen = max(maxdoclen, len(docs[-1]))

        return docs, maxdoclen