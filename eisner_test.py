import unittest

import numpy as np

from conll import Sentence, Token
from decoders.eisner import eisner


class EisnerTestCase(unittest.TestCase):
    def test_oracle_tree(self):
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

if __name__ == '__main__':
    unittest.main()