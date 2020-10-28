import pandas as pd


class OpenIE(object):
    '''
    Class that represents the data triples from an Information Extraction tool.

    Attributes:
        triples (pandas.DataFrame): DataFrame contanining all triples with the following columns:
                                    sentence id, triple id (within sentence),
                                    arg0, relation, arg1
    '''

    def __init__(self, data):
        self.triples = data

    @classmethod
    def read_csv(cls, filename):
        '''
        Reads a CSV file containing all necessary information in specific collumns.
            - Sentence id (column 0)
            - Triple id (column 2)
            - Arg0 (column 3)
            - Relation (column 4)
            - Arg1 (column 5)

        Parameters:
            filename (str): file to be read

        Returns:
            OpenIE: corresponding object itinialized with the pandas.DataFrame
        '''
        df = pd.read_csv(filename, sep=';',
                         usecols=[0, 2, 3, 4, 5],
                         names=['sent_id', 'triple_id', 'arg0', 'rel', 'arg1'],
                         skiprows=1, encoding='utf-8')
        df['sent_id'] = df['sent_id'].ffill()
        df['words'] = (df['arg0'] + ' ' + df['rel'] + ' ' +  # Unique words in the whole triple
                       df['arg1']).fillna('').str.split()
        df['vocab'] = df['words'].apply(set)
        df['sent_len'] = df['words'].apply(len)
        return cls(df)

    def get_triples(self, sent_id):
        '''Return all triples whithin the sentence with the given sentence id.'''
        return self.triples.groupby(by='sent_id').get_group(sent_id)
