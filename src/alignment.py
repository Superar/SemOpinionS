import penman


class Alignment(object):
    '''
    Class that stores alignment information.
    Should be created using the methods `read_giza` or `read_jamr` according to the format.

    Attributes:
        sentences (list): all sentences from the original file in lower case
        alignment (list): list of dictionaries concept-to-word, parallel with sentences
        word_to_concept (list): list of dictionaries word-to-concept, parallel with sentences
    '''

    def __init__(self, sentences, alignment):
        self.sentences = sentences
        self.alignment = alignment

        self.word_to_concept = list()
        for sent in alignment:
            words = dict()
            for concept in sent:
                for w in sent[concept]:
                    if w not in words:
                        words[w] = list()
                    words[w].append(concept)
            self.word_to_concept.append(words)

    def __getitem__(self, item):
        '''Numerical indexing of alignments'''
        return self.alignment[item]

    @classmethod
    def read_giza(cls, filepath):
        '''
        Creates an Alignment object from reading an alignment file in GIZA format.

        Parameters:
            filepath (str): Path to the alignment file in GIZA format

        Returns:
            Alignment: Alignment object containing all alignments read
        '''
        sentences = list()
        alignment = list()
        with open(filepath, encoding='utf-8') as file_:
            for line in file_:
                if line.startswith('#'):
                    toks = [t.split('_')[0] for t in line.lstrip('#').split()]
                    sent = ' '.join(toks)
                    sentences.append(sent.lower())

                    cur_alignment = dict()
                    amr_string = file_.readline()
                    amr = penman.loads(amr_string)[0]

                    # Node alignments
                    alignment_list = penman.surface.alignments(amr)
                    for a in alignment_list:
                        if a[-1] not in cur_alignment:
                            cur_alignment[a[-1]] = list()
                        cur_alignment[a[-1]].append(
                            toks[alignment_list[a].indices[0]])

                    # Role alignments
                    alignment_list = penman.surface.role_alignments(amr)
                    for a in alignment_list:
                        s, l, t = a
                        if (s, t, l) not in cur_alignment:
                            cur_alignment[(s, t, l)] = list()
                        cur_alignment[(s, t, l)].append(
                            toks[alignment_list[a].indices[0]])

                    alignment.append(cur_alignment)
        return cls(sentences, alignment)

    @classmethod
    def read_jamr(cls, filepath):
        sentences = list()
        alignment = list()
        with open(filepath, encoding='utf-8') as file_:
            cur_toks = list()
            for line in file_:
                if line.startswith('# ::snt'):
                    cur_sent = line[len('# ::snt '):].rstrip()
                    sentences.append(cur_sent.lower())
                    alignment.append(dict())
                elif line.startswith('# ::tok'):
                    cur_toks = line[len('# ::tok '):].rstrip().split()
                elif line.startswith('# ::node'):
                    node_data = line[len('# ::node\t'):].rstrip().split('\t')
                    if len(node_data) > 2:
                        # There is an alignment for the node
                        if node_data[1] not in alignment[-1]:
                            alignment[-1][node_data[1]] = list()
                        start, end = map(int, node_data[2].split('-'))
                        for i in range(start, end):
                            alignment[-1][node_data[1]].append(cur_toks[i])
        return cls(sentences, alignment)

    def get_sentence_position(self, sentence):
        '''
        Given a sentence, return its index in all paralle attributes lists

        Parameters:
            sentence (str): Exact sentence from the alignment file
        
        Returns:
            Integer: Index of the sentence if it is found
            None: If the sentence has not been found
        '''
        sentence = sentence.lower()
        for i, sent in enumerate(self.sentences):
            if sentence in sent:
                return i
        return None

    def get_alignments(self, sentence):
        '''
        Given a sentence, return its concept-to-word dictionary

        Parameters:
            sentence (str): Exact sentence form the alignment file

        Returns:
            Dict: Alignment concept-to-word dictionary for the given sentence
            None: If the sentence has not been found
        '''
        idx = self.get_sentence_position(sentence)
        if idx is not None:
            return self.alignment[idx]
        else:
            return None

    def get_reverse_alignments(self, sentence):
        '''
        Given a sentence, return its word-to-concept dictionary

        Parameters:
            sentence (str): Exact sentence form the alignment file

        Returns:
            Dict: Alignment word-to-concept dictionary for the given sentence
            None: If the sentence has not been found
        '''
        idx = self.get_sentence_position(sentence)
        if idx is not None:
            return self.word_to_concept[idx]
        else:
            return None
