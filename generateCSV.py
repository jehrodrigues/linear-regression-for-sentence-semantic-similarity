# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from commons import read_xml
import csv

DATA_DIR = './data/'

def read_ASSIN_data():
    pairs_train = read_xml('%sassin-ptbr-dev.xml' % (DATA_DIR), True)
    with open('assin-dev.csv', 'wb') as csvfile:
        filewriter = csv.writer(csvfile, delimiter='\t')  # quotechar='|', quoting=csv.QUOTE_MINIMAL
        filewriter.writerow(['pair_ID', 'sentence_A', 'sentence_B', 'relatedness_score', 'SemEval_set'])  # , 'SemEval_set'
        for pair in pairs_train:
            #filewriter.writerow(['1', 'sentence 1', 'sentence 2', '4.9'])
            id = pair.id
            t = pair.t
            h = pair.h
            sim = pair.similarity
            print(id)
            print(t)
            print(h)
            print(sim)
            #filewriter.writerow(['1534', 'A mulher está picando alho', 'Os brócolis estão sendo picados por uma mulher', '3.0'])
            filewriter.writerow([id, t.encode('utf-8'), h.encode('utf-8'), round(sim, 2), 'DEV'])

def read_SICK_data(file_path):
    df = pd.read_csv('/home/jessica/Documents/Projects/ASSIN2/Siamese-Sentence-Similarity/data/SICK_translated.csv', sep=',')
    df = df[['pair_ID', 'sentence_A', 'sentence_B', 'relatedness_score', 'SemEval_set']]
    df['relatedness_score'] = df['relatedness_score'].map(str)
    for i, score in enumerate(df['relatedness_score'].values):
        if len(score) > 4:
            score_str = score[:-3]
            score_str = score_str[:1] + '.' + score_str[1:]
            df['relatedness_score'][i] = score_str
    #print(df['relatedness_score'])
    df['SemEval_set'] = np.where(df['SemEval_set'] == 'TRIAL', 'DEV', df['SemEval_set'])

    df.to_csv(file_path, sep='\t', encoding='utf-8')

read_SICK_data('/home/jessica/Documents/Projects/ASSIN2/Siamese-Sentence-Similarity/data/SICK_translated_ok.csv')
