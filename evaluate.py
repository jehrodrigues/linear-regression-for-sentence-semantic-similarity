"""
This script evaluates a embedding model in a semantic similarity perspective.
It uses the dataset of ASSIN sentence similarity shared task and the method
of Hartmann which achieved the best results in the competition.
ASSIN shared-task website:
http://propor2016.di.fc.ul.pt/?page_id=381
Paper of Hartmann can be found at:
http://www.linguamatica.com/index.php/linguamatica/article/download/v8n2-6/365
"""

from commons import read_xml
from assin_eval import eval_similarity
from os import path
import argparse

DATA_DIR = './data/'
TEST_DIR = path.join(DATA_DIR, 'assin-test-gold/')


if __name__ == '__main__':
    # Parser descriptors
    parser = argparse.ArgumentParser(
        description='''Sentence similarity evaluation for word embeddings in
        brazilian and european variants of Portuguese language. It is expected
        a word embedding model in text format.''')

    parser.add_argument('lang',
                        choices=['br', 'pt'],
                        help='{br, eu} choose PT-BR or PT-EU testset')

    args = parser.parse_args()
    lang = args.lang
    #emb = args.embedding


    # Evaluating
    pairs_gold = read_xml('%sassin-pt%s-test.xml' % (TEST_DIR, lang), True)
    pairs_sys = read_xml('%soutput_first.xml' % DATA_DIR, True)
    #eval_rte(pairs_gold, pairs_sys)
    eval_similarity(pairs_gold, pairs_sys)

#python evaluate.py br