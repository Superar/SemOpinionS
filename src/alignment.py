import penman


class Alignment(object):
    def __init__(self, sentences, alignment):
        self.sentences = sentences
        self.alignment = alignment

    def __getitem__(self, item):
        return self.alignment[item]

    @classmethod
    def read_giza(cls, filepath):
        sentences = list()
        alignment = list()
        sent_num = 1
        with open(filepath, encoding='utf-8') as file_:
            for line in file_:
                if line.startswith('#'):
                    toks = [t.split('_')[0] for t in line.lstrip('#').split()]
                    sent = ' '.join(toks)
                    sentences.append(sent.lower())

                    alignment.append(dict())
                    amr_string = file_.readline()
                    alignment_list = penman.surface.alignments(
                        penman.loads(amr_string)[0])
                    for a in alignment_list:
                        if a[-1] not in alignment[-1]:
                            alignment[-1][a[-1]] = list()
                        alignment[-1][a[-1]].append(toks[alignment_list[a].indices[0]])
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
                    sentences.append(cur_sent)
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
        for i, sent in enumerate(self.sentences):
            if sent == sentence:
                return i
        return None

    def get_alignments(self, sentence):
        idx = self.get_sentence_position(sentence)
        if idx:
            return self.alignment[idx]
        else:
            return None
