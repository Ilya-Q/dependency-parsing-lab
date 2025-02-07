from typing import Optional, NamedTuple, List
from functools import cached_property
import dataclasses

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
        
ROOT = Token(id=0, pos="__ROOTPOS__", form="__ROOTFORM__")

class Sentence(List[Token]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.insert(0, ROOT)

    def write(self):
        return "\n".join(token.write() for token in self[1:]) + "\n"

    @cached_property
    def no_syntax(self):
        return Sentence([token._replace(head=None, rel=None) for token in self[1:]])

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

@dataclasses.dataclass
class Eval:
    total: int = dataclasses.field(default=0)
    true_arcs: int = dataclasses.field(default=0)
    true_labels: int = dataclasses.field(default=0)

    def compare(self, gold: Sentence, pred: Sentence, ignore_label=True) -> bool:
        assert(len(gold) == len(pred))
        ok = True
        for gold_tok, pred_tok in zip(gold[1:], pred[1:]):
            self.total += 1
            if gold_tok.head == pred_tok.head:
                self.true_arcs += 1
                if gold_tok.rel == pred_tok.rel:
                    self.true_labels += 1
                elif not ignore_label:
                    ok = False
            else:
                ok = False
        return ok

    def UAS(self) -> float:
        return self.true_arcs/self.total
    def LAS(self) -> float:
        return self.true_labels/self.total