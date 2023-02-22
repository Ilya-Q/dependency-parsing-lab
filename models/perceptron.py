import random
from typing import Callable, Sequence, Set, Tuple
from functools import partial
from itertools import chain
from conll import Sentence, Token, Eval
from features import TemplateFeatures
from tqdm import tqdm

class StructuredPerceptron:
    def __init__(self, decoder: Callable[[Sentence, Callable[[Token, Token], float]], Sentence]):
        self._w = {}
        self._decoder = decoder

    def fit(self, train_data: Sequence[TemplateFeatures], epochs=10):
        for ep in range(epochs):
            eval = Eval()
            random.shuffle(train_data)
            progress = tqdm(train_data, desc=f"Epoch={ep}")
            for features in progress:
                pred_sent = self.parse(features)
                if not eval.compare(features.sent, pred_sent):
                    for feature in chain.from_iterable(features.gold):
                        if feature not in self._w:
                            self._w[feature] = 1
                        else:
                            self._w[feature] += 1
                    for feature in chain.from_iterable(features(pred_sent[dep.head], dep) for dep in pred_sent[1:]):
                        if feature not in self._w:
                            self._w[feature] = -1
                        else:
                            self._w[feature] -= 1
                progress.set_postfix_str(f"UAS={eval.UAS():.4f}, n_weights={len(self._w)}")
            progress.close()

    def score_arc(self, features: TemplateFeatures, head: Token, dep: Token):
        return sum(self._w.get(feature, 0) for feature in features(head, dep))
    
    def parse(self, features: TemplateFeatures):
        return self._decoder(features.sent.no_syntax, partial(self.score_arc, features))