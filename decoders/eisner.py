from copy import deepcopy
from typing import Callable

import numpy as np

from conll import Sentence, Token

def eisner(sent: Sentence, score: Callable[[Token, Token], float]) -> Sentence:
    n = len(sent)
    Or = np.full((n, n), -np.inf)
    Or[np.diag_indices(n)] = 0.0
    Ol, Cr, Cl = Or.copy(), Or.copy(), Or.copy()
    Or_b, Ol_b, Cr_b, Cl_b = [np.empty((n,n), dtype=np.int64) for _ in range(4)]
    for m in range(1, n+1):
        for s in range(n-m):
            t = s+m
            Or[s, t], Or_b[s, t] = max(((Cl[s, q] + Cr[q+1, t] + score(sent[t], sent[s]), q) for q in range(s, t)), key=lambda t: t[0])
            Ol[s, t], Ol_b[s, t] = max(((Cl[s, q] + Cr[q+1, t] + score(sent[s], sent[t]), q) for q in range(s, t)), key=lambda t: t[0])
            Cr[s, t], Cr_b[s, t] = max(((Cr[s, q] + Or[q, t], q) for q in range(s, t)), key=lambda t: t[0])
            Cl[s, t], Cl_b[s, t] = max(((Ol[s, q] + Cl[q, t], q) for q in range(s+1, t+1)), key=lambda t: t[0])
    ret = [None] * n
    def recover_right_dependents(s, t):
        if s == t:
            return
        q = Cl_b[s, t]
        recover_right_dependents(q, t)
        ret[q] = sent[q]._replace(head=s)
        t, q = q, Ol_b[s, q]
        recover_right_dependents(s, q)
        recover_left_dependents(q+1, t)
    
    def recover_left_dependents(s, t):
        if s == t:
            return
        q = Cr_b[s, t]
        recover_right_dependents(s, q)
        ret[q] = sent[q]._replace(head=t)
        s = q
        q = Or_b[q, t]
        recover_right_dependents(s, q)
        recover_left_dependents(q+1, t)

    recover_right_dependents(0, n-1)
    return Sentence(ret[1:])