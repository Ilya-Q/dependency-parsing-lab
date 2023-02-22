from typing import Callable
from conll import Sentence, Token
import numpy as np
from copy import copy

def is_tree(sent: Sentence):
    owned_tokens = {0}
    for token in sent[1:]:
        if token.id in owned_tokens:
            continue
        chain = {token.id}
        while token.id not in owned_tokens:
            if token.head is None:
                # unattached token
                return False
            token = sent[token.head]
            if token.id in chain:
                # a cycle
                return False
            chain.add(token.id)
        owned_tokens.update(chain)
    return len(owned_tokens) == len(sent)

def score_sent(sent: Sentence, score: Callable[[Token, Token], float]):
    return sum(score(sent[dep.head], dep) for dep in sent[1:])

def npa(sent: Sentence, score: Callable[[Token, Token], float]) -> Sentence:
    # so we don't mess up the caller's copy
    # a shallow copy is enough because the tokens are immutable
    sent = copy(sent)
    best_score = score_sent(sent, score)
    while True:
        m, c, p = -np.inf, -1, -1
        for i in range(1, len(sent)):
            for j in range(len(sent)):
                hypot = copy(sent)
                hypot[i] = hypot[i]._replace(head=j)
                if not is_tree(hypot):
                    continue
                s = score_sent(hypot, score)
                if s > m:
                    m, c, p = s, i, j
        if m - best_score > 0:
            best_score = m
            sent[c] = sent[c]._replace(head=p)
        else:
            return sent
