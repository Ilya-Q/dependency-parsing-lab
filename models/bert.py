import transformers
import torch
from conll import Sentence, Token
from functools import partial

class BertParser(torch.nn.Module):
    def __init__(self, decoder, backbone='bert-base-uncased') -> None:
        super().__init__()
        self._decoder = decoder
        self.backbone = transformers.BertModel.from_pretrained(backbone)
        self.hidden = torch.nn.Linear(self.backbone.config.hidden_size * 2, 512)
        self.output = torch.nn.Linear(512, 1)

    def forward(self, tokenized: dict, sentence: Sentence):
        bert_out = self.backbone(tokenized["input_ids"])
        # only keep the first subtoken of every original token
        # TODO: replace with some kind of attention or RNN?
        embeddings = bert_out.last_hidden_state[tokenized["offset_mapping"][:,:,0] == 0] 
        embeddings = embeddings[:-1,:] # remove the [SEP] token
        # arc_embeddings = torch.vstack([torch.cat((head, dep)) for head_i, head in enumerate(embeddings) for dep_i, dep in enumerate(embeddings) if dep_i != 0 and head_i != dep_i])
        # arc_scores = self.output(torch.tanh(self.hidden(arc_embeddings)))
        self._cache = {}
        gold_arcs = {(dep.head, dep.id) for dep in sentence[1:]}
        score = partial(self.score_arc, embeddings)
        best_parse = self._decoder(sentence.no_syntax, score)
        penalty = sum(1.0 for dep in best_parse[1:] if (dep.head, dep.id) not in gold_arcs)
        best_parse_scores = torch.cat([score(best_parse[dep.head], dep) for dep in best_parse[1:]])
        gold_parse_scores = torch.cat([score(sentence[dep.head], dep) for dep in sentence[1:]])
        print(best_parse_scores)
        print(gold_parse_scores)
        loss = torch.sum(best_parse_scores) + torch.tensor(penalty) - torch.sum(gold_parse_scores) + torch.tensor(1.0)
        loss = torch.clamp(loss, min=0)
        del self._cache
        return best_parse, loss

    def score_arc(self, embeddings: torch.Tensor, head: Token, dep: Token):
        if (head.id, dep.id) in self._cache:
            return self._cache[(head.id, dep.id)]
        arc_embedding = torch.cat((embeddings[head.id], embeddings[dep.id]))
        score = self.output(torch.tanh(self.hidden(arc_embedding)))
        self._cache[(head.id, dep.id)] = score
        return score
