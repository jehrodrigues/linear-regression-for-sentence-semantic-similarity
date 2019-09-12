"""
This script evaluates a embedding model in a semantic similarity perspective.
It uses the dataset of ASSIN sentence similarity shared task and the method
of Hartmann which achieved the best results in the competition.
ASSIN shared-task website:
http://propor2016.di.fc.ul.pt/?page_id=381
Paper of Hartmann can be found at:
http://www.linguamatica.com/index.php/linguamatica/article/download/v8n2-6/365
"""

from gensim.models import FastText
from sklearn.linear_model import LinearRegression
from commons import read_xml
from assin_eval import eval_similarity
from assin_eval import eval_rte
from gensim.models import KeyedVectors
from xml.dom import minidom
from numpy import array
from os import path
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import pickle
import argparse
import unicodedata
import nlpnet

DATA_DIR = './data/'
TEST_DIR = path.join(DATA_DIR, 'assin-test-gold/')

tagger = nlpnet.POSTagger('/home/jessica/Documents/Projects/wordembeddings/assin/data/pos-pt', language='pt')

def set_part_of_speech_tag(sent):
    sent = str(sent).replace("[", "")
    sent = sent.replace("]", "")
    sent = sent.replace("\'", "")
    sent = sent.replace(",", "")
    tags = tagger.tag(str(sent))
    tags = str(tags).replace("\', \'", "|")
    tags = tags.replace("(", "")
    tags = tags.replace(")", "")
    tags = tags.replace("[[", "")
    tags = tags.replace("]]", "")
    tags = tags.replace(",", "")
    tags = tags.replace("\'", "")
    return tags.split()


def set_part_of_speech(word):
    tag = tagger.tag(str(word.lower()))[0][0][1]
    return str(word) + '|' + tag


def gensim_embedding_difference(data): #, field1, field2
    """Calculate the similarity between the sum of all embeddings."""
    distances = []
    for pair in data:
        #print("t: ", normalize_terms(pair.t.lower()))
        e1 = [set_part_of_speech(i) if set_part_of_speech(i) in embeddings else 'maqueros|N' for i in normalize_terms(pair.t.lower())]
        e2 = [set_part_of_speech(i) if set_part_of_speech(i) in embeddings else 'maqueros|N' for i in normalize_terms(pair.h.lower())]
        distances.append([embeddings.n_similarity(e1, e2)])
    return distances


def evaluate_testset(x, y, test):
    """Docstring."""
    l_reg = LinearRegression()
    l_reg.fit(x, y)
    test_predict = l_reg.predict(test)
    return test_predict


def write_xml(filename, pred):
    """Docstring."""
    with open(filename) as fp:
        xml = minidom.parse(fp)
    pairs = xml.getElementsByTagName('pair')
    for pair in pairs:
        pair.setAttribute('similarity', str(pred[pairs.index(pair)]))
    with open(filename, 'w') as fp:
        fp.write(xml.toxml())

def normalize_terms(terms):
    # Remove Numerals
    #terms = remove_numerals(terms)

    # Remove Punctuation and tokenize
    terms = remove_punctuation(terms)

    # Remove StopWords
    filtered_words = [word for word in terms if word not in stopwords.words('portuguese')]

    # Remove Accents
    #filtered_words = [remove_accents(term).lower() for term in terms]

    # Stemming
    #st = nltk.stem.RSLPStemmer()
    #st = nltk.stem.SnowballStemmer('portuguese')
    #filtered_stem = [st.stem(term) for term in terms]
    #filtered_stem = [st.stem(filtered_word) for filtered_word in terms]

    return filtered_words

def remove_punctuation(term):
    """Remove Punctuation and tokenize"""
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(term)


if __name__ == '__main__':
    # Parser descriptors
    parser = argparse.ArgumentParser(
        description='''Sentence similarity evaluation for word embeddings in
        brazilian and european variants of Portuguese language. It is expected
        a word embedding model in text format.''')

    parser.add_argument('embedding',
                        type=str,
                        help='embedding model')

    parser.add_argument('lang',
                        choices=['br', 'pt'],
                        help='{br, eu} choose PT-BR or PT-EU testset')

    args = parser.parse_args()
    lang = args.lang
    emb = args.embedding

    # Loading embedding model
    embeddings = KeyedVectors.load_word2vec_format(emb,
                                                   binary=False,
                                                   unicode_errors="ignore")

    pairs_train = read_xml('%sassin2-train.xml' % (DATA_DIR), True)

    pairs_test = read_xml('%sassin-ptbr-test.xml' % (TEST_DIR), True)


    # Loading evaluation data and parsing it
    #with open('%sassin-pt%s-train.pkl' % (DATA_DIR, lang), 'rb') as fp:
        #data = pickle.load(fp)

    #with open('%sassin-pt%s-test-gold.pkl' % (DATA_DIR, lang), 'rb') as fp:
        #test = pickle.load(fp)

    # Getting features
    #features = gensim_embedding_difference(data, 'tokens_t1', 'tokens_t2')
    features = gensim_embedding_difference(pairs_train)
    #features_test = gensim_embedding_difference(test, 'tokens_t1', 'tokens_t2')
    features_test = gensim_embedding_difference(pairs_test)

    # Predicting similarities
    #results = array([float(i['result']) for i in data])

    results = array([float(i.similarity) for i in pairs_train])
    results_test = evaluate_testset(features, results, features_test)

    write_xml('%soutput.xml' % DATA_DIR, results_test)

    # Evaluating
    pairs_gold = read_xml('%sassin-pt%s-test.xml' % (TEST_DIR, lang), True)
    pairs_sys = read_xml('%soutput.xml' % DATA_DIR, True)
    #eval_rte(pairs_gold, pairs_sys)
    eval_similarity(pairs_gold, pairs_sys)

#python evaluate_assin2_sense.py ./models/sense2vec_s300_ptbreu_sg.txt br