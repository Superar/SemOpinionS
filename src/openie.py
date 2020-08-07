import pandas as pd
import numpy as np


class OpenIE(object):
    def __init__(self, data):
        self.triples = data

    @classmethod
    def read_csv(cls, filename):
        df = pd.read_csv(filename, sep=';',
                         usecols=[0, 2, 3, 4, 5],
                         names=['sent_id', 'triple_id', 'arg0', 'rel', 'arg1'],
                         skiprows=1, encoding='utf-8')
        df['sent_id'] = df['sent_id'].ffill()
        df['words'] = (df['arg0'] + ' ' + df['rel'] + ' ' + # Unique words in the whole triple
                       df['arg1']).fillna('').str.split()
        df['vocab'] = df['words'].apply(set)
        df['sent_len'] = df['words'].apply(len)
        return cls(df)

    def get_triples(self, sent_id):
        return self.triples.groupby(by='sent_id').get_group(sent_id)
