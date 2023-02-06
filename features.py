from conll import Sentence, Token
import re

def _parse_template(template: str):
    return [(t, p, int(o) if len(o) > 0 else None)
           for t, p, o in re.findall(r"([hdb])(form|pos)([+-]\d+)?", template)
    ]

DEFAULT_TEMPLATES = {t: _parse_template(t) for t in [
    # unigram features
    "hform,hpos",
    "hform",
    "hpos",
    "dform,dpos",
    "dform",
    "dpos",
    # bigram features
    "hform,hpos,dform,dpos",
    "hpos,dform,dpos",
    "hform,dform,dpos",
    "hform,hpos,dform",
    "hform,hpos,dpos",
    "hform,dform",
    "hpos,dpos",
    # other
    "hpos,bpos,dpos",
    "hpos,dpos,hpos+1,dpos-1",
    "hpos,dpos,hpos-1,dpos-1",
    "hpos,dpos,hpos+1,dpos+1",
    "hpos,dpos,hpos-1,dpos+1"
]}

class TemplateFeatures:
    def __init__(self, sent: Sentence, templates=DEFAULT_TEMPLATES, cache=True):
        self._sent = sent
        self._cache = {} if cache else None
        self._templates = templates

    def __call__(self, head: Token, dep: Token):
        if self._cache and (head, dep) in self._cache:
            return self._cache[(head, dep)]
        # TODO: abstract individual templates out?
        dist = head.id - dep.id
        direction = "L"
        if dist < 0:
            dist = -dist
            direction = "R"
        feats = set()
        for name, spec in self._templates.items():
            bslot = None
            values = []
            for target, prop, offset in spec:
                if target == 'b':
                    assert(bslot is None)
                    bslot = len(values)
                    values.append(prop)
                    continue
                elif target == 'h':
                    target_idx = head.id
                elif target == 'd':
                    target_idx = dep.id
                else:
                    raise ValueError(f"Unknown template target type '{target}'")
                if offset is not None:
                    target_idx += offset
                try:
                    token = self._sent[target_idx]
                except IndexError:
                    values.append("__NULL__")
                    continue
                if prop == 'pos':
                    values.append(token.pos)
                elif prop == 'form':
                    values.append(token.form)
                else:
                    raise ValueError(f"Unknown token property '{prop}")
            if bslot is None:
                feats.add(f"{name}={'+'.join(values)}+{direction}+{dist}")
            else:
                prop = values[bslot]
                if dist == 1:
                    values[bslot] = "__NULL__"
                    feats.add(f"{name}={'+'.join(values)}+{direction}+{dist}")
                else:
                    base = dep.id if direction == 'L' else head.id
                    for idx in range(base+1, base+dist):
                        if prop == 'pos':
                            values[bslot] = self._sent[idx].pos
                        elif prop == 'form':
                            values[bslot] = self._sent[idx].form
                        else:
                            raise ValueError(f"Unknown token property '{prop}")
                        feats.add(f"{name}={'+'.join(values)}+{direction}+{dist}")
        if self._cache:
            self._cache[(head, dep)] = feats
        return feats