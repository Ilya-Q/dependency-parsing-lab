import unittest

import numpy as np

from conll import Sentence, Token, read_conll
from decoders.eisner import eisner
from functools import partial
from tools import decode_parallel

class OracleScorer:
    def __init__(self, sent: Sentence):
        self.sent = sent
    def __call__(self, head, dep):
        return 100 if self.sent[dep.id].head == head.id else 1

class EisnerTestCase(unittest.TestCase):
    def test_example_tree(self):
        sent = Sentence([
            Token(id=1, form="John"),
            Token(id=2, form="saw"),
            Token(id=3, form="Mary")
        ])
        scores = {
            (0, 1): 9,
            (0, 2): 10,
            (0, 3): 9,
            (1, 2): 20,
            (1, 3): 3,
            (2, 1): 30,
            (2, 3): 30,
            (3, 1): 11,
            (3, 2): 0
        }
        parse = eisner(
            sent,
            lambda head, dep: float(
                scores.get((head.id, dep.id), -np.inf)
            )
        )
        self.assertEqual(len(sent), len(parse))
        self.assertEqual(parse[1].head, 2),
        self.assertEqual(parse[2].head, 0),
        self.assertEqual(parse[3].head, 2)

    def test_oracle_trees(self):
        sents = read_conll("data/english/train/wsj_train.only-projective.conll06")
        parses = partial(decode_parallel, eisner)((sent.strip_syntax(), OracleScorer(sent)) for sent in sents)
        for gold_sent, parse in zip(sents, parses):
            with self.subTest(sent=gold_sent):
                for gold_token, parsed_token in zip(gold_sent, parse):
                    self.assertEqual(gold_token.head, parsed_token.head)
if __name__ == '__main__':
    unittest.main()