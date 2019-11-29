import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import pickle
import spacy
import scipy.sparse
from scipy.sparse import csr_matrix
import math
from sklearn.metrics.pairwise import linear_kernel
nlp=spacy.load('en_core_web_lg')


""" Tokenizing"""
def _keep_token(t):
    return (t.is_alpha and
            not (t.is_space or t.is_punct or
                 t.is_stop or t.like_num))
def _lemmatize_doc(doc):
    return [ t.lemma_ for t in doc if _keep_token(t)]

def _preprocess(doc_list):
    return [_lemmatize_doc(nlp(doc)) for doc in doc_list]
def dummy_fun(doc):
    return doc

# Importing List of 128.000 Metadescriptions:
Web_data=open("./data/meta_descriptions","r", encoding="utf-8")
All_lines=Web_data.readlines()
# outputs a list of meta-descriptions consisting of lists of preprocessed tokens:
data=_preprocess(All_lines)

# TF-IDF Vectorizer:
vectorizer = TfidfVectorizer(min_df=10,tokenizer=dummy_fun,preprocessor=dummy_fun,)
    tfidf = vectorizer.fit_transform(data)
dictionary = vectorizer.get_feature_names()

# Retrieving Word embedding vectors:
temp_array=[nlp(dictionary[i]).vector for i in range(len(dictionary))]

# I had to build the sparse array in several steps due to RAM constraints
# (with bigrams the vocabulary gets as large as >1m
dict_emb_sparse=scipy.sparse.csr_matrix(temp_array[0])
for arr in range(1,len(temp_array),100000):
    print(str(arr))
    dict_emb_sparse=scipy.sparse.vstack([dict_emb_sparse, scipy.sparse.csr_matrix(temp_array[arr:min(arr+100000,len(temp_array))])])

# Multiplying the TF-IDF matrix with the Word embeddings:
tfidf_emb_sparse=tfidf.dot(dict_emb_sparse)

# Translating the Query into the TF-IDF matrix and multiplying with the same Word Embeddings:
query_doc= vectorizer.transform(_preprocess(["World of Books is one of the largest online sellers of second-hand books in the world Our massive collection of over million cheap used books also comes with free delivery in the UK Whether it s the latest book release fiction or non-fiction we have what you are looking for"]))
query_emb_sparse=query_doc.dot(dict_emb_sparse)

# Calculating Cosine Similarities:
cosine_similarities = linear_kernel(query_emb_sparse, tfidf_emb_sparse).flatten()

related_docs_indices = cosine_similarities.argsort()[:-10:-1]

# Printing the Site descriptions with the highest match:
for ID in related_docs_indices:
    print(All_lines[ID])