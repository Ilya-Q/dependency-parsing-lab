from features import TemplateFeatures
from conll import read_conll, write_conll, Eval
from decoders.npa import npa
from pathlib import Path
from functools import partial
from tqdm import tqdm
import pickle
import lzma
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_file", "-m", type=Path)
parser.add_argument("--preds", "-p", type=Path)
parser.add_argument("--npa", action="store_true", default=False)
group = parser.add_mutually_exclusive_group()
group.add_argument("--gold", "-g", type=read_conll)
group.add_argument("--blind", "-b", type=read_conll)

args = parser.parse_args()

with lzma.open(args.model_file, 'r') as f:
    model = pickle.load(f)

out = []
if args.gold is not None:
    eval = Eval()
    for sent in tqdm(args.gold):
        feats = TemplateFeatures(sent)
        pred_sent = model.parse(feats)
        if args.npa:
            pred_sent = npa(pred_sent, score=partial(model.score_arc, feats))
        out.append(pred_sent)
        eval.compare(sent, pred_sent)
elif args.blind is not None:
    for sent in args.blind:
        feats = TemplateFeatures(sent, cache_all=True)
        pred_sent = model.parse(feats)
        out.append(pred_sent)
else:
    raise ValueError("Either the gold or the blind file is required")

if args.preds is not None:
    write_conll(args.preds, out)
if args.gold is not None:
    print(f"UAS={eval.UAS()}")