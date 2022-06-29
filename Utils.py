import pandas as pd
import numpy as np 
import math
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

import matplotlib.pyplot as plt



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


def top_rec(title_of_movie, similarity_matrix, indices, df,n):
    """
    Given a cosine siilarity matrix and item to id mapping, this function returns the top 10 items with the highest similarity.
    
    Input: 
    title_of_movie (String): title of movie
    similarity_matrix (2D array): similarity matrix
    indices (DataFrame): Pandas DF, title of movies as index and values are ID
    
    Return:
    top (Series): Pandas Series, Top movies with highest cosine score
    
    """

    movie_indices = [i[0] for i in sorted(list(enumerate(similarity_matrix[indices[title_of_movie]])), key=lambda x: x[1], reverse=True)[1:n]]
    
    top  = df['title'].iloc[movie_indices]
    return top






def top_rec_test(user, similarity_matrix, indices, df, rating_test, n):
    
    titles = rating_test[rating_test.userId == user].sort_values('rating', ascending = False).movieId.values[0:2]
    
    if len(titles) == 2:
        title1 = df[df.movieId == titles[0]].title.values[0]
        title2 = df[df.movieId == titles[1]].title.values[0]
        
        movie_indices1 = [i[0] for i in sorted(list(enumerate(similarity_matrix[indices[title1]])), key=lambda x: x[1], reverse=True)[1:int(n/2)+1]]

        top1  = df[['title','movieId']].iloc[movie_indices1]


        movie_indices2 = [i[0] for i in sorted(list(enumerate(similarity_matrix[indices[title2]])), key=lambda x: x[1], reverse=True)[1:math.ceil(n/2)+1]]

        top2  = df[['title','movieId']].iloc[movie_indices2]

        top = pd.concat([top1, top2])
    
    
    else:
        
        title1 = df[df.movieId == titles[0]].title.values[0]

        movie_indices1 = [i[0] for i in sorted(list(enumerate(similarity_matrix[indices[title1]])), key=lambda x: x[1], reverse=True)[1:n+1]]

        top  = df[['title','movieId']].iloc[movie_indices1]


    
    return top

def top_rec_user(userId, sim, k,n,df, user_movie, indexes ):
    
    # K is for how many users to average
    # n is for how many movies to recommend
    # indexes is user index to id
    
    user = indexes[indexes.userId == userId].index.values[0]
    temp = user_movie
    users_ord = [i[0] for i in sorted(list(enumerate(sim[user])), key=lambda x: x[1], reverse=True)[1:k]]
    temp.iloc[users_ord].mean(axis = 0)
    temp.reset_index(drop = True,inplace = True)
    movie_indices = temp.iloc[users_ord].mean(axis = 0).sort_values(ascending = False).index[0:n].to_numpy().flatten()  
    
    return df.iloc[movie_indices]


def evaluate(rating_test, n, test_type, df, sample_size = 200, rating = None, sim = None, user_matrix = None, user_movie = None, indexes = None, k = 10):
    ##rating_test,rating, sim, n, test_type,df, user_matrix = None):
    
    if test_type == 'simple':
        users = random_sample(rating_test.userId.unique(), sample_size)
        hits = 0
        for user in users:
            preds = simple(user,df,rating,n).movieId
            targets = rating_test[(rating_test.userId == user) & (rating_test.rating > 3.5)].movieId
            hits = hits + targets[targets.isin(preds)].count()

        return hits/len(users)
    
    if test_type == 'CB':     
        users = random_sample(rating_test.userId.unique(), sample_size)
        hits = 0
        for user in users:
            preds = top_rec_test(user, sim, pd.Series(df.index, index=df['title']), df, rating_test, n).movieId
            targets = rating_test[(rating_test.userId == user) & (rating_test.rating > 3.5)].movieId
            hits = hits + targets[targets.isin(preds)].count()
        return hits/len(users)  
            
    if test_type == 'UB':  
        
        users = random_sample(rating_test.userId.unique(), sample_size)
        hits = 0
        for user in users:
            preds = top_rec_user(user, sim, k,n,df, user_matrix, indexes).movieId
            targets = rating_test[(rating_test.userId == user) & (rating_test.rating > 3.5)].movieId
            hits = hits + targets[targets.isin(preds)].count() 
        return hits/len(users)
            
    if test_type == 'Hybrid':
        c = ['1st Max','2nd Max','3rd Max', '4th Max', '5th Max', '6th Max', '7th Max','8th Max', '9th Max','10th Max']
        temp = (user_movie.apply(lambda x: pd.Series(x.nlargest(10).index, index=c), axis=1)
            .reset_index())
        
        users = random_sample(rating_test.userId.unique(), sample_size)
        hits = 0
        for user in users:
            preds = CB_CF_MEMORY_HYBRID_REC_test(temp, user, df, n).movieId
            targets = rating_test[(rating_test.userId == user) & (rating_test.rating > 3.5)].movieId
            hits = hits + targets[targets.isin(preds)].count()  
            
        return hits/len(users)


############################################
# Simple models
############################################

def simple(user,df,rating, n):
    return df[~df.movieId.isin(rating[rating.userId == user].movieId )].sort_values('score', ascending=False).head(n)

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




def Rating2Vec(user_item):
    return cosine_similarity(user_item, user_item)



def CB_CF_MEMORY_HYBRID_REC(user_movie, user, df, n):
    
    c = ['1st Max','2nd Max','3rd Max', '4th Max', '5th Max', '6th Max', '7th Max','8th Max', '9th Max','10th Max']
    temp = (user_movie.apply(lambda x: pd.Series(x.nlargest(10).index, index=c), axis=1)
            .reset_index())

    indx = temp[temp.userId == user][['1st Max','2nd Max','3rd Max',
                                         '4th Max', '5th Max', '6th Max', '7th Max','8th Max', '9th Max','10th Max']].values[0]

    return df[['title','movieId']].iloc[indx][0:n]


def CB_CF_MEMORY_HYBRID_REC_test( top_matrix, user, df, n):

    indx = top_matrix[top_matrix.userId == user][['1st Max','2nd Max','3rd Max',
                                         '4th Max', '5th Max', '6th Max', '7th Max','8th Max', '9th Max','10th Max']].values[0]

    return df[['title','movieId']].iloc[indx][0:n]





def utils_plot_keras_training(training):
    metrics = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15,3))
    
    ## training
    ax[0].set(title="Training")
    ax11 = ax[0].twinx()
    ax[0].plot(training.history['loss'], color='black')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax11.plot(training.history[metric], label=metric)
    ax11.set_ylabel("Score", color='steelblue')
    ax11.legend()
    
    ## validation
    ax[1].set(title="Validation")
    ax22 = ax[1].twinx()
    ax[1].plot(training.history['val_loss'], color='black')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax22.plot(training.history['val_'+metric], label=metric)
    ax22.set_ylabel("Score", color="steelblue")
    plt.show()
    
    
def NN_Test(test,n = 5):
    tp = 0
    
    for user in range(test.user.nunique()):
        top = n
        y_test = test[test["user"]==user].sort_values("y", ascending=False)["product"].values[:top]

        predicted = test[test["user"]==user].sort_values("yhat", ascending=False)["product"].values[:top]
        
        tp += len(list(set(y_test) & set(predicted)))
    
    return tp/test.user.nunique()    
    
    
    
    
    