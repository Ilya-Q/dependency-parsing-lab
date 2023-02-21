from features import TemplateFeatures
from conll import read_conll, write_conll, Eval
from pathlib import Path
import pickle
import lzma
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gold", "-g", type=read_conll)
parser.add_argument("--model_file", "-m", type=Path)
parser.add_argument("--preds", "-p", type=Path)
args = parser.parse_args()

with lzma.open(args.model_file, 'r') as f:
    model = pickle.load(f)

out = []
eval = Eval()
for sent in args.gold:
    feats = TemplateFeatures(sent)
    pred_sent = model.parse(feats)
    out.append(pred_sent)
    eval.compare(sent, pred_sent)

if args.preds is not None:
    write_conll(args.preds, out)
print(f"UAS={eval.UAS()}")