import penman


class Alignment(object):
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
        return self.alignment[item]

    @classmethod
    def read_giza(cls, filepath):
        sentences = list()
        alignment = list()
        with open(filepath, encoding='utf-8') as file_:
            for line in file_:
                if line.startswith('#'):
                    toks = [t.split('_')[0] for t in line.lstrip('#').split()]
                    sent = ' '.join(toks)
                    sentences.append(sent.lower())

                    alignment.append(dict())
                    amr_string = file_.readline()
                    amr = penman.loads(amr_string)[0]

                    # Node alignments
                    alignment_list = penman.surface.alignments(amr)
                    for a in alignment_list:
                        if a[-1] not in alignment[-1]:
                            alignment[-1][a[-1]] = list()
                        alignment[-1][a[-1]
                                      ].append(toks[alignment_list[a].indices[0]])

                    # Role alignments
                    alignment_list = penman.surface.role_alignments(amr)
                    for a in alignment_list:
                        s, l, t = a
                        if (s, t, l) not in alignment[-1]:
                            alignment[-1][(s, t, l)] = list()
                        alignment[-1][(s, t, l)
                                      ].append(toks[alignment_list[a].indices[0]])
        return cls(sentences, alignment)

    @classmethod
    def read_jamr(cls, filepath):
        sentences = list()
        alignment = list()
        with open(filepath, encoding='utf-8') as file_:
            cur_toks = list()
            for line in file_:
                if line.startswith('# ::snt'):
                    cur_sent = line.lstrip('# ::snt ').rstrip()
                    sentences.append(cur_sent.lower())
                    alignment.append(dict())
                elif line.startswith('# ::tok'):
                    cur_toks = line.lstrip('# ::tok ').rstrip().split()
                elif line.startswith('# ::node'):
                    node_data = line.lstrip('# ::node\t').rstrip().split('\t')
                    if node_data[1] not in alignment[-1]:
                        alignment[-1][node_data[1]] = list()

                    if len(node_data) > 2:
                        # There is an alignment for the node
                        start, end = map(int, node_data[2].split('-'))
                        for i in range(start, end):
                            alignment[-1][node_data[1]].append(cur_toks[i])
        return cls(sentences, alignment)

    def get_sentence_position(self, sentence):
        sentence = sentence.lower()
        for i, sent in enumerate(self.sentences):
            if sent == sentence:
                return i
        return None

    def get_alignments(self, sentence):
        idx = self.get_sentence_position(sentence)
        if idx is not None:
            return self.alignment[idx]
        else:
            return None

    def get_reverse_alignments(self, sentence):
        idx = self.get_sentence_position(sentence)
        if idx is not None:
            return self.word_to_concept[idx]
        else:
            return None
