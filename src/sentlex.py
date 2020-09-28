class SentimentLexicon(object):
    def __init__(self, word_to_sent):
        self.word_to_sent = word_to_sent

    def __iter__(self):
        return iter(self.word_to_sent)

    def __getitem__(self, item):
        return self.word_to_sent[item]

    @classmethod
    def read_oplexicon(cls, filepath):
        word_to_sent = dict()
        with open(filepath, 'r', encoding='utf-8') as file_:
            for line in file_:
                w, _, s, _ = line.split(',')
                word_to_sent[w] = int(s)
        return cls(word_to_sent)
