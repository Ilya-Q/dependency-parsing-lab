from typing import Optional, NamedTuple, List

class Token(NamedTuple):
    id: int
    form: Optional[str] = None
    lemma: Optional[str] = None
    pos: Optional[str] = None
    xpos: Optional[str] = None
    morph: Optional[str] = None
    head: Optional[int] = None
    rel: Optional[str] = None

    @classmethod
    def read(cls, line: str):
        fields = [ field if field != '_' else None for field in line.split('\t', maxsplit=10)[:8]]
        fields[0] = int(fields[0])
        if fields[6] is not None:
            fields[6] = int(fields[6])
        return cls(*fields)

    def write(self):
        fields = [str(field) if field is not None else "_" for field in self] + ["_", "_"]
        return "\t".join(fields)
        
ROOT = Token(id=0)

class Sentence(List[Token]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.insert(0, ROOT)

    def write(self):
        return "\n".join(token.write() for token in self[1:]) + "\n"

def read_conll(path: str) -> List[Sentence]:
    sentences = []
    with open(path, "r", encoding='utf-8') as f:
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

def write_conll(path: str, sents: List[Sentence]):
    with open(path, 'w', encoding='utf-8') as f:
        for sent in sents:
            raw = sent.write()
            f.write(raw)
            f.write('\n')


            