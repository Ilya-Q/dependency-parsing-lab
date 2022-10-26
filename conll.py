from typing import Optional, NamedTuple, List

class Token(NamedTuple):
    id: int
    form: str
    lemma: str
    pos: str
    xpos: str
    morph: str
    head: Optional[int]
    rel: str

    @classmethod
    def read(cls, line: str):
        fields = line.split('\t', maxsplit=10)
        id, form, lemma, pos, xpos, morph, head, rel = fields[:8]
        id = int(id)
        if head == '_':
            head = None
        else:
            head = int(head)
        return cls(id, form, lemma, pos, xpos, morph, head, rel)

    def write(self):
        fields = [str(field) if field is not None else "_" for field in self] + ["_", "_"]
        return "\t".join(fields)
        
        
class Sentence(List[Token]):
    def write(self):
        return "\n".join(token.write() for token in self) + "\n"

def read_conll(file):
    sentences = []
    with open(file, "r", encoding='utf-8') as f:
        while True:
            sent = []
            raw = f.readline()
            if raw == '':
                break
            while raw != '\n':
                sent.append(Token.read(raw))
                raw = f.readline()
            sentences.append(Sentence(sent))
    return sentences

def write_conll(file, sents):
    with open(file, 'w', encoding='utf-8') as f:
        for sent in sents:
            raw = sent.write()
            f.write(raw)
            f.write('\n')


            