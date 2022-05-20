import pandas as pd
import numpy as np 

from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk

import string

import keras
from keras.preprocessing.text import one_hot,Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense , Flatten ,Embedding,Input
from keras.models import Model

import re

from gensim.corpora import Dictionary
from gensim.similarities import WordEmbeddingSimilarityIndex
from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix
from gensim.models import Word2Vec

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity


# Importing modules
import os

# LDA Model
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
from pprint import pprint
from gensim.models import CoherenceModel
import spacy
# Import the wordcloud library
#from wordcloud import WordCloud
# Visualize the topics
import pyLDAvis.gensim_models
import pickle 
import pyLDAvis





############################################
# Generic Functions
############################################

def random_sample(arr: np.array, size: int = 10000) -> np.array:
    """Samples from a Numpy array"""
    return arr[np.random.choice(len(arr), size=size, replace=False)]


def clean_text(text, toke = False):
    """
    Cleans a string of text
    
    Input: 
    Text (String): Text to be cleaned
    toke (Boolean): If True, function returns text in tokens.
    
    Returns:
    text (String): Cleaned text
    
    """
    default_stemmer = PorterStemmer()
    default_stopwords = stopwords.words('english')
    
    
    def tokenize_text(text):
        return [w for s in sent_tokenize(text) for w in word_tokenize(s)]

        

    def remove_special_characters(text, characters=string.punctuation.replace('-', '')):
        tokens = tokenize_text(text)
        pattern = re.compile('[{}]'.format(re.escape(characters)))
        return ' '.join(filter(None, [pattern.sub('', t) for t in tokens]))

    def stem_text(text, stemmer=default_stemmer):
        tokens = tokenize_text(text)
        return ' '.join([stemmer.stem(t) for t in tokens])

    def remove_stopwords(text, stop_words=default_stopwords):
        tokens = [w for w in tokenize_text(text) if w not in stop_words]
        return ' '.join(tokens)

    text = text.strip(' ') # strip whitespaces
    text = text.lower() # lowercase
    text = stem_text(text) # stemming
    text = remove_special_characters(text) # remove punctuation and symbols
    text = remove_stopwords(text) # remove stopwords
    if toke == True:
        text = tokenize_text(text)
    #text.strip(' ') # strip whitespaces again?

    return text


def top_rec(title_of_movie, similarity_matrix, indices, df):
    """
    Given a cosine siilarity matrix and item to id mapping, this function returns the top 10 items with the highest similarity.
    
    Input: 
    title_of_movie (String): title of movie
    similarity_matrix (2D array): similarity matrix
    indices (DataFrame): Pandas DF, title of movies as index and values are ID
    
    Return:
    top (Series): Pandas Series, Top movies with highest cosine score
    
    """

    movie_indices = [i[0] for i in sorted(list(enumerate(similarity_matrix[indices[title_of_movie]])), key=lambda x: x[1], reverse=True)[1:11]]
    
    top  = df['title'].iloc[movie_indices]
    return top







############################################
# Content Based models
############################################

def TF_IDF(column, ngram = 2):
    
    tf = TfidfVectorizer(analyzer='word',ngram_range=(1, ngram),min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(column)
    return linear_kernel(tfidf_matrix, tfidf_matrix)
    

def Word2Vec_Hybrid(text_token, vector_size = 300, window = 7, epochs = 25):

    corpus_sentences = text_token

    model = Word2Vec(vector_size=vector_size, window=window, min_count=5, workers=11, alpha=0.025)

    model.build_vocab(corpus_sentences)

    model.train(corpus_sentences, total_examples=model.corpus_count, epochs=epochs)
    
    termsim_index = WordEmbeddingSimilarityIndex(model.wv) # similartiy between word embeddings


    corpus_list_token = text_token.tolist() # list of tokenized sentences

    dictionary = Dictionary(corpus_list_token) #word id and word

    bow_corpus = [dictionary.doc2bow(document) for document in corpus_list_token] # word id and frequency for each doc

    similarity_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary)

    
    # a method that allows us to assess the similarity between two documents in a meaningful way, even when they have no words in common. It uses a measure of similarity between words
    docsim_index = SoftCosineSimilarity(bow_corpus, similarity_matrix)

    return docsim_index[bow_corpus]



def Doc2Word_embed(text_clean, text_token, vector_size = 300, window = 7,epochs = 25):
    corpus_sentences = text_token

    model = Word2Vec(vector_size=vector_size, window=window, min_count=5, workers=11, alpha=0.025)

    model.build_vocab(corpus_sentences)

    model.train(corpus_sentences, total_examples=model.corpus_count, epochs=epochs)

    word_em = model.wv.vectors

    embeddings_dict = {}
    for word in model.wv.index_to_key:
        embeddings_dict[word] = model.wv[word]

    def doc2vecF(doc):
        vdoc = [embeddings_dict.get(x,0) for x in doc.lower().split(" ")]
        doc2vec = np.sum(vdoc, axis = 0)
        if np.sum(doc2vec == 0) ==1:
            print("hi")
            doc2vec = np.zeros(50, "float32")

        return doc2vec

    data = []
    for i in text_clean:
        data.append(doc2vecF(i))
    embd = pd.DataFrame(data)

    return cosine_similarity(embd, embd)




def Rating2Vec(user_item_pivot_df):
    return cosine_similarity(user_item_pivot_df, user_item_pivot_df)
