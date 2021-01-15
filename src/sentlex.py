from pathlib import Path

class SentimentLexicon(object):
    """
    Class that reads a Sentiment Lexicon file.

    Attributes:
        word_to_sent: dict (word -> sentiment)
    """

    def __init__(self, word_to_sent: dict):
        self.word_to_sent = word_to_sent

    def __iter__(self):
        return iter(self.word_to_sent)

    def __getitem__(self, item: str):
        return self.word_to_sent[item]

    @classmethod
    def read_oplexicon(cls, filepath: Path):
        """
        Creates an object from a file in the OpLexicon fromat.
        
        Parameters:
            filepath (Path): Path of the file to be read

        Return:
            SentimentLexicon: An object with all sentiment mappings
        """
        word_to_sent = dict()
        with open(filepath, 'r', encoding='utf-8') as file_:
            for line in file_:
                w, _, s, _ = line.split(',')
                word_to_sent[w] = int(s)
        return cls(word_to_sent)
